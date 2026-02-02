#!/usr/bin/env python3
"""
Calculate roughness statistics for labeled point clouds.

Computes roughness for Ground (label 1) and Stem (label 2) across all scans,
then reports aggregate statistics: median, IQR, and 95th percentile.

Configuration is loaded from roughness_config.yaml in the same directory.
"""

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from processing.calc_geom_feature import GeometricFeatureCalculator, create_xyz_spatial_batches
from sklearn.neighbors import KDTree


def load_config(config_path='roughness_config.yaml'):
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Please create roughness_config.yaml in the pcd_profiler directory"
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand home directory in paths
    if 'data_root' in config:
        config['data_root'] = Path(config['data_root']).expanduser()
    if 'output_dir' in config and config['output_dir']:
        config['output_dir'] = Path(config['output_dir']).expanduser()
    
    return config


def _read_labels(label_path: Path) -> np.ndarray:
    """Read labels from file."""
    labels = pd.read_csv(label_path, header=None, sep=r'\s+', dtype=np.int32).values
    labels = labels.squeeze().astype(np.int32)
    return labels

def read_csv_and_labels(csv_path, label_path):
    """Read point cloud CSV and corresponding label file."""
    df = pd.read_csv(csv_path)
    
    # Extract XYZ coordinates
    if all(col in df.columns for col in ['X', 'Y', 'Z']):
        xyz_cols = ['X', 'Y', 'Z']
    elif all(col in df.columns for col in ['x', 'y', 'z']):
        xyz_cols = ['x', 'y', 'z']
    else:
        raise ValueError(f"Could not find XYZ columns in {csv_path}")
    
    points_xyz = df[xyz_cols].values
    labels = _read_labels(label_path)
    
    if len(labels) != len(points_xyz):
        raise ValueError(
            f"Mismatch: {len(points_xyz)} points vs {len(labels)} labels in {csv_path.name}"
        )
    
    return points_xyz, labels


def calculate_roughness_gpu(points_xyz, config):
    """Calculate roughness using GPU acceleration with automatic batching for large clouds."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    
    num_points = len(points_xyz)
    neighbor_radius = config['neighbor_radius']
    max_neighbors = config['max_neighbors']
    batch_size = config['batch_size']
    buffer_pts = config['buffer_pts']
    
    # Small point clouds: direct processing
    if num_points <= batch_size:
        calculator = GeometricFeatureCalculator(device='cuda')
        calculator.set_points_on_gpu(points_xyz)
        
        geom_features = calculator.estimate_geometric_features_batched(
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors
        )
        
        roughness = geom_features['roughness']
        del calculator
        torch.cuda.empty_cache()
        return roughness
    
    # Large point clouds: batch processing with buffer
    print(f"  Large cloud detected ({num_points:,} pts), using batch processing...")
    
    points_df = pd.DataFrame(points_xyz, columns=['X', 'Y', 'Z'])
    batches = create_xyz_spatial_batches(points_df, pts_num_per_batch=batch_size, max_batch_size=batch_size)
    print(f"  Created {len(batches)} batches (avg: {num_points//len(batches):,} pts/batch)")
    
    kdtree = KDTree(points_xyz)
    roughness_full = np.zeros(num_points, dtype=np.float32)
    calculator = GeometricFeatureCalculator(device='cuda')
    
    for batch_idx, batch_df in enumerate(batches):
        core_indices = batch_df.index.to_numpy()
        core_points = batch_df[['X', 'Y', 'Z']].to_numpy()
        
        neighbor_indices = kdtree.query(core_points, k=buffer_pts, return_distance=False)
        buffered_indices = np.unique(np.concatenate([neighbor_indices.flatten(), core_indices]))
        buffered_points = points_xyz[buffered_indices]
        core_mask = np.isin(buffered_indices, core_indices)
        
        calculator.set_points_on_gpu(buffered_points)
        geom_features = calculator.estimate_geometric_features_batched(
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors
        )
        
        roughness_full[core_indices] = geom_features['roughness'][core_mask]
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(batches)} batches")
    
    del calculator
    torch.cuda.empty_cache()
    return roughness_full


def process_scan(csv_path, label_path, config):
    """Process a single scan: read data, calculate roughness, filter by label."""
    points_xyz, labels = read_csv_and_labels(csv_path, label_path)
    
    unique_labels = np.unique(labels)
    print(f"  Points: {len(points_xyz):,} | Labels: {unique_labels}")
    
    # Calculate roughness (in meters)
    roughness_m = calculate_roughness_gpu(points_xyz, config)
    roughness_cm = roughness_m * 100  # Convert to cm
    
    # Check for NaN values (but don't replace them yet)
    nan_mask = np.isnan(roughness_cm)
    nan_count = nan_mask.sum()
    valid_mask = ~nan_mask
    
    if nan_count > 0:
        print(f"  Warning: {nan_count:,}/{len(roughness_cm):,} points with NaN roughness (will be excluded)")
    
    # Process all labels dynamically
    results = {}
    label_info = []
    
    for label in unique_labels:
        label_mask = (labels == label) & valid_mask
        label_nan = ((labels == label) & nan_mask).sum()
        
        roughness_values = roughness_cm[label_mask]
        results[int(label)] = {
            'roughness': roughness_values,
            'nan_count': int(label_nan)
        }
        
        label_info.append(f"Label {label}: {len(roughness_values):,} valid ({label_nan:,} NaN)")
    
    print("  " + " | ".join(label_info))
    
    return results


def calculate_statistics(roughness_values):
    """Calculate median, IQR, and 95th percentile."""
    if len(roughness_values) == 0:
        return {'median': np.nan, 'iqr': np.nan, 'p95': np.nan, 'q25': np.nan, 'q75': np.nan}
    
    median = np.median(roughness_values)
    q25 = np.percentile(roughness_values, 25)
    q75 = np.percentile(roughness_values, 75)
    iqr = q75 - q25
    p95 = np.percentile(roughness_values, 95)
    
    return {'median': median, 'iqr': iqr, 'p95': p95, 'q25': q25, 'q75': q75}


def save_per_scan_results(output_dir, scan_name, label_stats):
    """Save per-scan roughness statistics to text file."""
    label_names = {
        1: 'Ground & Water',
        2: 'Stem',
        3: 'Canopy',
        4: 'Roots',
        5: 'Objects'
    }
    
    output_path = output_dir / f"{scan_name}_roughness.txt"
    
    with open(output_path, 'w') as f:
        f.write(f"Roughness Statistics: {scan_name}\n")
        f.write("=" * 60 + "\n\n")
        
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            label_name = label_names.get(label, f'Label {label}')
            f.write(f"{label_name} (Label {label}):\n")
            f.write(f"  Median:       {stats['median']:.2f} cm\n")
            f.write(f"  IQR:          {stats['iqr']:.2f} cm\n")
            f.write(f"  95th %ile:    {stats['p95']:.2f} cm\n")
            f.write(f"  Q25-Q75:      {stats['q25']:.2f} - {stats['q75']:.2f} cm\n\n")


def find_all_scan_files(data_root):
    """Find all CSV and label file pairs recursively."""
    scan_pairs = []
    
    # Find all CSV files recursively
    csv_files = sorted(data_root.rglob('*.csv'))
    
    for csv_path in csv_files:
        # Construct expected label path (same directory structure, pcd->label)
        label_path = Path(str(csv_path).replace('/pcd/', '/label/'))
        label_path = label_path.parent / f"{csv_path.stem.replace('_color', '_refined')}.label"
        
        if label_path.exists():
            scan_pairs.append((csv_path, label_path))
    
    return scan_pairs


def main():
    # Label name mapping
    label_names = {
        1: 'Ground & Water',
        2: 'Stem',
        3: 'Canopy',
        4: 'Roots',
        5: 'Objects'
    }
    
    # Load configuration
    config = load_config()
    data_root = config['data_root']
    
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")
    
    print("=" * 80)
    print("ROUGHNESS STATISTICS BY LABEL")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Neighbor radius: {config['neighbor_radius']}m")
    print(f"Max neighbors: {config['max_neighbors']}")
    print(f"Batch size: {config['batch_size']:,} points")
    if config.get('max_scans'):
        print(f"Max scans: {config['max_scans']} (testing mode)")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
    print("=" * 80)
    
    # Find all scan files
    scan_pairs = find_all_scan_files(data_root)
    
    if not scan_pairs:
        print("\n❌ No scan files found!")
        return
    
    # Limit scans if specified
    if config.get('max_scans'):
        scan_pairs = scan_pairs[:config['max_scans']]
    
    print(f"\nFound {len(scan_pairs)} scans to process\n")
    
    # Create output directory if specified
    output_dir = config.get('output_dir')
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Intermediate results will be saved to: {output_dir}\n")
    
    # Process all scans
    all_labels_data = {}  # Dictionary: {label: [roughness_arrays]}
    all_labels_nan = {}   # Dictionary: {label: nan_count}
    
    for csv_path, label_path in tqdm(scan_pairs, desc="Processing scans"):
        print(f"\n{csv_path.name}")
        
        scan_results = process_scan(csv_path, label_path, config)
        
        # Accumulate results for all labels
        for label, data in scan_results.items():
            if label not in all_labels_data:
                all_labels_data[label] = []
                all_labels_nan[label] = 0
            
            if len(data['roughness']) > 0:
                all_labels_data[label].append(data['roughness'])
            all_labels_nan[label] += data['nan_count']
        
        # Save per-scan statistics if output_dir is specified
        if output_dir:
            label_stats = {}
            for label, data in scan_results.items():
                label_stats[label] = calculate_statistics(data['roughness'])
            save_per_scan_results(output_dir, csv_path.stem, label_stats)
                
    
    # Combine results
    print(f"\n{'=' * 80}")
    print("COMBINING RESULTS FROM ALL SCANS")
    print(f"{'=' * 80}\n")
    
    if not all_labels_data:
        print("❌ No data processed successfully!")
        return
    
    # Concatenate arrays for each label
    label_roughness_all = {}
    for label, roughness_list in all_labels_data.items():
        if roughness_list:
            label_roughness_all[label] = np.concatenate(roughness_list)
        else:
            label_roughness_all[label] = np.array([])
    
    # Calculate totals
    total_valid_points = sum(len(arr) for arr in label_roughness_all.values())
    total_nan_points = sum(all_labels_nan.values())
    total_all_points = total_valid_points + total_nan_points
    
    print(f"Total points (all labels): {total_all_points:,} ({total_valid_points:,} valid, {total_nan_points:,} NaN)")
    print(f"\nPoints per label:")
    for label in sorted(label_roughness_all.keys()):
        print(f"  Label {label}: {len(label_roughness_all[label]):,} valid ({all_labels_nan.get(label, 0):,} NaN)")
    
    # Calculate statistics for all labels
    label_stats = {}
    for label, roughness_arr in label_roughness_all.items():
        label_stats[label] = calculate_statistics(roughness_arr)
    
    # Print results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS (across all scans)")
    print(f"{'=' * 80}\n")
    
    for label in sorted(label_stats.keys()):
        stats = label_stats[label]
        roughness_arr = label_roughness_all[label]
        nan_count = all_labels_nan.get(label, 0)
        label_name = label_names.get(label, f'Label {label}')
        
        print(f"{label_name} (Label {label}):")
        print(f"  Median Roughness:    {stats['median']:.2f} cm")
        print(f"  IQR:                 {stats['iqr']:.2f} cm")
        print(f"  95th Percentile:     {stats['p95']:.2f} cm")
        print(f"  Q25-Q75:             {stats['q25']:.2f} - {stats['q75']:.2f} cm")
        print(f"  Total points:        {len(roughness_arr):,} (excluded {nan_count:,} NaN)")
        print()
    
    print(f"\n{'=' * 80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    
    # Save results
    output_filename = config.get('output_file', 'roughness_statistics.txt')
    if output_dir:
        results_path = output_dir / output_filename
    else:
        results_path = Path(output_filename)
    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ROUGHNESS STATISTICS BY LABEL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data root: {data_root}\n")
        f.write(f"Neighbor radius: {config['neighbor_radius']}m\n")
        f.write(f"Max neighbors: {config['max_neighbors']}\n")
        f.write(f"Scans processed: {len(scan_pairs)}\n\n")
        f.write("=" * 80 + "\n")
        
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            roughness_arr = label_roughness_all[label]
            nan_count = all_labels_nan.get(label, 0)
            label_name = label_names.get(label, f'Label {label}')
            
            f.write(f"{label_name} (Label {label}):\n")
            f.write(f"  Median Roughness:    {stats['median']:.2f} cm\n")
            f.write(f"  IQR:                 {stats['iqr']:.2f} cm\n")
            f.write(f"  95th Percentile:     {stats['p95']:.2f} cm\n")
            f.write(f"  Q25-Q75:             {stats['q25']:.2f} - {stats['q75']:.2f} cm\n")
            f.write(f"  Total points:        {len(roughness_arr):,} (excluded {nan_count:,} NaN)\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Results saved to: {results_path.absolute()}\n")


if __name__ == "__main__":
    main()
