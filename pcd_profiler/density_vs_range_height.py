"""
Compute and visualize point cloud density vs. range from scanner.

This script analyzes point cloud density as a function of distance from the scan center,
which is useful for:
- Understanding scan geometry and occlusion behavior
- Detecting abnormal sparsity or dropouts
- Confirming expected density decay with range

Supports two modes:
1. Single mode: Analyze one scan (when csv_path is provided)
2. Aggregate mode: Analyze multiple scans (when csv_dir is provided)

Configuration is specified in density_vs_range_config.yaml
"""

import numpy as np
from pathlib import Path
import glob

from density_compute import compute_range_from_center, compute_density_vs_range, \
                            compute_aggregate_density_vs_range, \
                            compute_vertical_density_distribution, \
                            compute_vertical_density_distribution_aggregate
from density_io_plot import read_point_cloud_csv, plot_density_vs_range, \
                           plot_aggregate_density_vs_range, \
                           plot_vertical_density_violin, \
                           plot_vertical_density_violin_aggregate, load_config, \
                           save_results, load_results


def main():
    """
    Main function to run density vs range analysis using YAML configuration.
    """
    # Load configuration
    config = load_config()
    
    # Extract parameters from config
    csv_path = config.get('csv_path')
    csv_dir = config.get('csv_dir')
    x_col = config['columns']['x']
    y_col = config['columns']['y']
    z_col = config['columns']['z']
    scan_center = config.get('scan_center')
    
    bin_width = config['analysis']['bin_width']
    density_type = config['analysis']['density_type']
    k_neighbors = config['analysis'].get('k_neighbors')
    merge_threshold = config['analysis'].get('merge_threshold', 15.0)
    bin_height = config['analysis'].get('bin_height', 0.5)  # For vertical distribution
    
    output_dir = config['visualization'].get('output_dir')
    dpi = config['visualization'].get('dpi', 300)
    figsize = tuple(config['visualization'].get('figsize', [12, 6]))
    
    print(f"\n{'='*60}")
    print("Point Cloud Density vs Range Analysis")
    print(f"{'='*60}\n")
    
    # Determine mode: aggregate if csv_dir is provided and not null
    if csv_dir and csv_dir.strip():
        # AGGREGATE MODE
        print(f"Mode: AGGREGATE (processing multiple scans from directory)")
        print(f"Directory: {csv_dir}\n")
        
        # Find all CSV files in directory
        # csv_files = sorted(glob.glob(str(Path(csv_dir) / "*.csv")))
        csv_files = sorted(Path(csv_dir).rglob("*.csv"))
        
        if not csv_files:
            print(f"ERROR: No CSV files found in {csv_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load all scans
        points_list = []
        ranges_list = []
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n--- Loading scan {i}/{len(csv_files)}: {Path(csv_file).name} ---")
            points, _ = read_point_cloud_csv(csv_file, x_col, y_col, z_col)
            ranges, _ = compute_range_from_center(points, scan_center)
            
            points_list.append(points)
            ranges_list.append(ranges)
        
        # Compute aggregate statistics
        bin_centers, densities_median, densities_q25, densities_q75, \
        point_counts, scan_densities, scan_bin_centers = \
            compute_aggregate_density_vs_range(
                points_list, ranges_list, bin_width, density_type,
                k_neighbors, merge_threshold
            )
        
        # Print aggregate statistics
        valid_densities = densities_median[~np.isnan(densities_median)]
        if len(valid_densities) > 0:
            unit = 'points/m³' if density_type == 'volumetric' else 'points/m²'
            print(f"\nAggregate density statistics ({unit}):")
            print(f"  Median: {np.median(valid_densities):.2f}")
            print(f"  Q25: {np.percentile(valid_densities, 25):.2f}")
            print(f"  Q75: {np.percentile(valid_densities, 75):.2f}")
            print(f"  Min: {valid_densities.min():.2f}")
            print(f"  Max: {valid_densities.max():.2f}")
        
        # Plot aggregate
        plot_aggregate_density_vs_range(
            bin_centers, densities_median, densities_q25, densities_q75,
            point_counts, scan_densities, scan_bin_centers,
            density_type, output_dir, figsize, dpi
        )
        
        # Save density vs range results
        if output_dir:
            results_path = Path(output_dir) / 'results'
            save_results(
                results_path,
                mode='aggregate',
                bin_centers=bin_centers,
                densities_median=densities_median,
                densities_q25=densities_q25,
                densities_q75=densities_q75,
                point_counts=point_counts,
                scan_densities=scan_densities,
                scan_bin_centers=scan_bin_centers,
                density_type=density_type
            )
        
        # Compute and plot vertical density distribution (aggregate)
        print("\n" + "="*60)
        print("Computing vertical density distribution (aggregate)...")
        print("="*60)
        
        scan_heights, scan_densities, scan_ids, z_range = \
            compute_vertical_density_distribution_aggregate(
                points_list, bin_height, density_type, k_neighbors
            )
        
        # Generate output path for aggregate violin plot
        violin_output_dir = None
        if output_dir:
            violin_output_dir = output_dir
        
        plot_vertical_density_violin_aggregate(scan_heights, scan_densities, scan_ids, z_range,
                                              density_type, violin_output_dir,
                                              figsize, dpi)
        
        # Save vertical density results
        if output_dir:
            results_path = Path(output_dir) / 'results'
            save_results(
                results_path,
                mode='aggregate',
                scan_heights=scan_heights,
                scan_densities_vertical=scan_densities,
                scan_ids=scan_ids,
                z_range=z_range,
                density_type=density_type,
                bin_centers=bin_centers,
                densities_median=densities_median,
                densities_q25=densities_q25,
                densities_q75=densities_q75,
                point_counts=point_counts,
                scan_densities=scan_densities,
                scan_bin_centers=scan_bin_centers
            )
        
    else:
        # SINGLE MODE
        if not csv_path:
            print("ERROR: Either csv_path or csv_dir must be specified in config")
            return
        
        print(f"Mode: SINGLE (processing one scan)")
        print(f"File: {csv_path}\n")
        
        # Read point cloud
        points, df = read_point_cloud_csv(csv_path, x_col, y_col, z_col)
        
        # Compute ranges
        ranges, scan_center = compute_range_from_center(points, scan_center)
        print(f"Range statistics: min={ranges.min():.2f}m, max={ranges.max():.2f}m, "
              f"mean={ranges.mean():.2f}m")
        
        # Compute density vs range
        bin_centers, densities_median, densities_q25, densities_q75, point_counts = \
            compute_density_vs_range(points, ranges, bin_width, density_type, k_neighbors, merge_threshold)
        
        # Print statistics
        valid_densities = densities_median[~np.isnan(densities_median)]
        if len(valid_densities) > 0:
            unit = 'points/m³' if density_type == 'volumetric' else 'points/m²'
            print(f"\nDensity statistics ({unit}):")
            print(f"  Median: {np.median(valid_densities):.2f}")
            print(f"  Q25: {np.percentile(valid_densities, 25):.2f}")
            print(f"  Q75: {np.percentile(valid_densities, 75):.2f}")
            print(f"  Min: {valid_densities.min():.2f}")
            print(f"  Max: {valid_densities.max():.2f}")
        
        # Plot
        plot_density_vs_range(bin_centers, densities_median, densities_q25, densities_q75,
                             point_counts, density_type, output_dir,
                             figsize, dpi)
        
        # Save results
        if output_dir:
            results_path = Path(output_dir) / 'results'
            save_results(
                results_path,
                mode='single',
                bin_centers=bin_centers,
                densities_median=densities_median,
                densities_q25=densities_q25,
                densities_q75=densities_q75,
                point_counts=point_counts,
                density_type=density_type
            )
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
