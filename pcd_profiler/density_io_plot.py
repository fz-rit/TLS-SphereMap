"""
I/O and visualization functions for point cloud density analysis.

This module contains functions for:
- Reading point cloud data from CSV
- Loading configuration files
- Plotting density vs range results
- Plotting vertical density distribution (violin plots)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path
import yaml
import json


def read_point_cloud_csv(csv_path, x_col='x', y_col='y', z_col='z'):
    """
    Read point cloud from CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file containing point cloud data
    x_col, y_col, z_col : str
        Column names for x, y, z coordinates
        
    Returns:
    --------
    points : np.ndarray
        N x 3 array of point coordinates
    df : pd.DataFrame
        Full dataframe for additional processing
    """
    df = pd.read_csv(csv_path)
    
    # Check if columns exist
    if not all(col in df.columns for col in [x_col, y_col, z_col]):
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Could not find columns {x_col}, {y_col}, {z_col} in CSV")
    
    points = df[[x_col, y_col, z_col]].values
    print(f"Loaded {len(points)} points from {csv_path}")
    
    return points, df


def load_config(config_path='density_config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            f"Please create a config file based on density_vs_range_config.yaml"
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_file}")
    return config


def save_results(output_path, mode='aggregate', **data):
    """
    Save intermediate computation results to files.
    
    Saves numerical arrays to .npz format and metadata to .json format.
    This allows for quick replotting without recomputation.
    
    Parameters:
    -----------
    output_path : str or Path
        Base path for output files (without extension)
    mode : str
        'aggregate' or 'single'
    **data : dict
        Dictionary of data to save
    """
    output_path = Path(output_path)
    
    if mode == 'aggregate':
        # Save density vs range data
        csv_path = output_path.parent / 'density_vs_range.csv'
        json_path = output_path.parent / 'density_vs_range_metadata.json'
        
        # Create dataframe for aggregate statistics
        df_aggregate = pd.DataFrame({
            'bin_centers': data['bin_centers'],
            'densities_median': data['densities_median'],
            'densities_q25': data['densities_q25'],
            'densities_q75': data['densities_q75'],
            'point_counts': data['point_counts']
        })
        df_aggregate.to_csv(csv_path, index=False)
        
        # Save per-scan data
        scan_csv_path = output_path.parent / 'density_vs_range_per_scan.csv'
        scan_rows = []
        for i, (scan_dens, scan_centers) in enumerate(zip(data['scan_densities'], data['scan_bin_centers'])):
            for center, dens in zip(scan_centers, scan_dens):
                scan_rows.append({
                    'scan_id': i,
                    'bin_center': center,
                    'density': dens
                })
        df_scans = pd.DataFrame(scan_rows)
        df_scans.to_csv(scan_csv_path, index=False)
        
        # Save metadata to JSON
        metadata = {
            'mode': mode,
            'density_type': data.get('density_type', 'volumetric'),
            'n_scans': len(data['scan_densities']),
            'files': {
                'aggregate': str(csv_path.name),
                'per_scan': str(scan_csv_path.name)
            }
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDensity vs range results saved to:")
        print(f"  Aggregate data: {csv_path}")
        print(f"  Per-scan data: {scan_csv_path}")
        print(f"  Metadata: {json_path}")
        
        # Save vertical density distribution data
        if 'scan_heights' in data:
            vertical_csv = output_path.parent / 'vertical_density.csv'
            vertical_json = output_path.parent / 'vertical_density_metadata.json'
            
            # Build rows for vertical density data
            vertical_rows = []
            for i, (heights, densities, ids) in enumerate(zip(
                data['scan_heights'], data['scan_densities_vertical'], data['scan_ids']
            )):
                scan_id = ids if np.isscalar(ids) else (ids[0] if len(ids) > 0 else i)
                for h, d in zip(heights, densities):
                    vertical_rows.append({
                        'scan_id': scan_id,
                        'height': h,
                        'density': d
                    })
            
            df_vertical = pd.DataFrame(vertical_rows)
            df_vertical.to_csv(vertical_csv, index=False)
            
            # Save vertical metadata
            vertical_metadata = {
                'mode': mode,
                'density_type': data.get('density_type', 'volumetric'),
                'n_scans': len(data['scan_heights']),
                'z_range': list(data['z_range']),
                'files': {
                    'vertical_density': str(vertical_csv.name)
                }
            }
            with open(vertical_json, 'w') as f:
                json.dump(vertical_metadata, f, indent=2)
            
            print(f"\nVertical density results saved to:")
            print(f"  Data: {vertical_csv}")
            print(f"  Metadata: {vertical_json}")
    
    elif mode == 'single':
        csv_path = output_path.parent / 'density_vs_range_single.csv'
        json_path = output_path.parent / 'density_vs_range_single_metadata.json'
        
        df = pd.DataFrame({
            'bin_centers': data['bin_centers'],
            'densities_median': data['densities_median'],
            'densities_q25': data['densities_q25'],
            'densities_q75': data['densities_q75'],
            'point_counts': data['point_counts']
        })
        df.to_csv(csv_path, index=False)
        
        metadata = {
            'mode': mode,
            'density_type': data.get('density_type', 'volumetric'),
            'files': {
                'density_data': str(csv_path.name)
            }
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  Data: {csv_path}")
        print(f"  Metadata: {json_path}")


def load_results(results_path):
    """
    Load intermediate computation results from files.
    
    Parameters:
    -----------
    results_path : str or Path
        Path to the .csv or .json file (either works)
        
    Returns:
    --------
    data : dict
        Dictionary containing loaded data
    metadata : dict
        Dictionary containing metadata
    """
    results_path = Path(results_path)
    
    # If given a CSV path, find the corresponding JSON metadata
    if results_path.suffix == '.csv':
        json_path = results_path.with_name(results_path.stem.replace('_per_scan', '').replace('_single', '') + '_metadata.json')
    else:
        json_path = results_path.with_suffix('.json')
    
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    data = {}
    
    if metadata['mode'] == 'aggregate':
        # Load density vs range aggregate data
        aggregate_csv = json_path.parent / metadata['files']['aggregate']
        if aggregate_csv.exists():
            df_aggregate = pd.read_csv(aggregate_csv)
            data['bin_centers'] = df_aggregate['bin_centers'].values
            data['densities_median'] = df_aggregate['densities_median'].values
            data['densities_q25'] = df_aggregate['densities_q25'].values
            data['densities_q75'] = df_aggregate['densities_q75'].values
            data['point_counts'] = df_aggregate['point_counts'].values
        
        # Load per-scan data
        scan_csv = json_path.parent / metadata['files']['per_scan']
        if scan_csv.exists():
            df_scans = pd.read_csv(scan_csv)
            scan_densities = []
            scan_bin_centers = []
            
            for scan_id in sorted(df_scans['scan_id'].unique()):
                scan_data = df_scans[df_scans['scan_id'] == scan_id]
                scan_densities.append(scan_data['density'].values)
                scan_bin_centers.append(scan_data['bin_center'].values)
            
            data['scan_densities'] = scan_densities
            data['scan_bin_centers'] = scan_bin_centers
        
        # Check for vertical density data
        vertical_json = json_path.parent / 'vertical_density_metadata.json'
        if vertical_json.exists():
            with open(vertical_json, 'r') as f:
                vertical_meta = json.load(f)
            
            vertical_csv = json_path.parent / vertical_meta['files']['vertical_density']
            if vertical_csv.exists():
                df_vertical = pd.read_csv(vertical_csv)
                
                scan_heights = []
                scan_densities_vertical = []
                scan_ids = []
                
                for scan_id in sorted(df_vertical['scan_id'].unique()):
                    scan_data = df_vertical[df_vertical['scan_id'] == scan_id]
                    scan_heights.append(scan_data['height'].values)
                    scan_densities_vertical.append(scan_data['density'].values)
                    scan_ids.append(scan_id)
                
                data['scan_heights'] = scan_heights
                data['scan_densities_vertical'] = scan_densities_vertical
                data['scan_ids'] = scan_ids
                data['z_range'] = vertical_meta['z_range']
    
    elif metadata['mode'] == 'single':
        csv_path = json_path.parent / metadata['files']['density_data']
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        data['bin_centers'] = df['bin_centers'].values
        data['densities_median'] = df['densities_median'].values
        data['densities_q25'] = df['densities_q25'].values
        data['densities_q75'] = df['densities_q75'].values
        data['point_counts'] = df['point_counts'].values
    
    print(f"\nLoaded results from: {json_path.parent}")
    return data, metadata


def plot_density_vs_range(bin_centers, densities_median, densities_q25, densities_q75, 
                         point_counts, density_type='volumetric', 
                         output_dir=None,
                         figsize=(10, 8), dpi=300):
    """
    Plot density vs range with IQR uncertainty bands.
    
    Parameters:
    -----------
    bin_centers : np.ndarray
        Center of each range bin
    densities_median : np.ndarray
        Median density in each bin
    densities_q25 : np.ndarray
        25th percentile of density
    densities_q75 : np.ndarray
        75th percentile of density
    point_counts : np.ndarray
        Number of points in each bin
    density_type : str
        'volumetric' or 'areal'
    output_dir : str or None
        Path to save figure. If None, displays plot.
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        DPI for saved figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Filter out NaN values
    valid_mask = ~np.isnan(densities_median)
    valid_centers = bin_centers[valid_mask]
    valid_densities = densities_median[valid_mask]
    valid_q25 = densities_q25[valid_mask]
    valid_q75 = densities_q75[valid_mask]
    
    # Main plot: Density vs Range (LOG SCALE)
    ax1.plot(valid_centers, valid_densities, 'o-', linewidth=2, 
             markersize=4, label='Median Density', color='steelblue')
    
    # Add IQR band
    ax1.fill_between(valid_centers, valid_q25, valid_q75,
                     alpha=0.3, color='steelblue', label='IQR (25-75%)')
    
    ax1.set_xlabel('Range from Scanner (m)', fontsize=12)
    unit = 'points/m³' if density_type == 'volumetric' else 'points/m²'
    ax1.set_ylabel(f'Point Density ({unit}, log scale)', fontsize=12)
    ax1.set_title('Point Cloud Density vs. Range', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both', linestyle=':')
    ax1.legend()
    
    # Secondary plot: Point count per bin (LOG SCALE)
    # Calculate bar widths (handle variable bin widths for merged bins)
    if len(bin_centers) > 1:
        bar_widths = np.diff(np.append(bin_centers - (bin_centers[1] - bin_centers[0])/2, 
                                       bin_centers[-1] + (bin_centers[-1] - bin_centers[-2])/2))
        bar_widths = bar_widths * 0.8  # Make bars slightly narrower
    else:
        bar_widths = [1.0]
    
    ax2.bar(bin_centers, point_counts, width=bar_widths,
            alpha=0.6, color='gray', edgecolor='black')
    ax2.set_xlabel('Range from Scanner (m)', fontsize=12)
    ax2.set_ylabel('Point Count (log scale)', fontsize=12)
    ax2.set_title('Points per Range Bin', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both', linestyle=':', axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'density_vs_range.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_aggregate_density_vs_range(bin_centers, densities_median, densities_q25, densities_q75,
                                    point_counts, scan_densities, scan_bin_centers,
                                    density_type='volumetric', output_dir=None,
                                    figsize=(8, 6), dpi=300):
    """
    Plot aggregated density vs range across multiple scans.
    
    Shows individual scan lines (colored), median across scans (thick black),
    and IQR band (shaded).
    
    Parameters:
    -----------
    bin_centers : np.ndarray
        Center of each range bin
    densities_median : np.ndarray
        Median of scan-level medians
    densities_q25 : np.ndarray
        25th percentile across scans
    densities_q75 : np.ndarray
        75th percentile across scans
    point_counts : np.ndarray
        Median point count per bin
    scan_densities : list of np.ndarray
        Per-scan median densities
    scan_bin_centers : list of np.ndarray
        Per-scan bin centers
    density_type : str
        'volumetric' or 'areal'
    output_dir : str or None
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved figure
    """
    # Set journal-friendly style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Main plot: Density vs Range
    
    # Get tab20 colormap for individual scans
    cmap = plt.cm.get_cmap('tab20')
    n_scans = len(scan_densities)
    
    # Plot individual scans with different colors
    for i, (scan_dens, scan_centers) in enumerate(zip(scan_densities, scan_bin_centers)):
        valid_mask = ~np.isnan(scan_dens)
        if np.any(valid_mask):
            color = cmap(i % 20)  # Cycle through 20 colors if more scans
            ax.plot(scan_centers[valid_mask], scan_dens[valid_mask],
                    '-', linewidth=1.2, alpha=0.7, color=color)
    
    # Filter out NaN values for aggregate
    valid_mask = ~np.isnan(densities_median)
    valid_centers = bin_centers[valid_mask]
    valid_densities = densities_median[valid_mask]
    valid_q25 = densities_q25[valid_mask]
    valid_q75 = densities_q75[valid_mask]
    
    # Plot IQR band (shaded)
    ax.fill_between(valid_centers, valid_q25, valid_q75,
                     alpha=0.3, color='steelblue', label='IQR across scans')
    
    # Plot median across scans as thick black line
    ax.plot(valid_centers, valid_densities, 'o-', linewidth=2.5,
             markersize=4, label='Median across scans', color='black')
    
    ax.set_xlabel('Range from Scanner (m)', fontsize=14, fontweight='bold')
    unit = 'points/m³' if density_type == 'volumetric' else 'points/m²'
    ax.set_ylabel(f'Point Density ({unit})', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.8)
    ax.tick_params(labelsize=11)
    
    # Create custom legend with rainbow bar for individual scans
    # Create a horizontal color bar showing scan colors
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    legend_elements = []
    
    # Add a rainbow bar representing individual scans
    # Create a single entry that shows multiple colors
    n_colors_to_show = min(n_scans, 20)
    box_width = 0.15 / n_colors_to_show  # Total width of ~0.15 inches
    
    # Create proxy for individual scans with gradient
    class RainbowHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            
            # Draw small rectangles with different colors
            n_boxes = min(n_scans, 10)  # Show up to 10 color boxes
            box_w = width / n_boxes
            patches = []
            for i in range(n_boxes):
                idx = int(i * n_scans / n_boxes)  # Sample colors evenly
                color = cmap(idx % 20)
                rect = Rectangle((x0 + i * box_w, y0), box_w, height,
                                facecolor=color, edgecolor='none', alpha=0.7)
                handlebox.add_artist(rect)
            return patches
    
    # Custom proxy object for individual scans
    individual_scans_proxy = Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor='gray', markersize=8)
    
    legend_elements = [
        (individual_scans_proxy, f'Individual scans'),
        (Line2D([0], [0], color='steelblue', alpha=0.5, linewidth=8), 'IQR across scans'),
        (Line2D([0], [0], color='black', linewidth=2.5, marker='o', markersize=4), 'Median across scans')
    ]
    
    # Create legend
    legend = ax.legend([el[0] for el in legend_elements], 
                      [el[1] for el in legend_elements],
                      handler_map={individual_scans_proxy: RainbowHandler()},
                      loc='upper right', frameon=True, framealpha=0.9,
                      edgecolor='black', fancybox=False, fontsize=14)
    legend.get_frame().set_linewidth(1.2)
    
    plt.tight_layout()
    
    if output_dir:
        # Create output directory if it doesn't exist
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir_path / 'density_vs_range.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_vertical_density_violin(local_densities, bin_labels, point_counts, 
                                density_type='volumetric', output_dir=None,
                                figsize=(12, 6), dpi=300):
    """
    Plot vertical density distribution as violin plots.
    
    Parameters:
    -----------
    local_densities : np.ndarray
        Local density values for all points
    bin_labels : np.ndarray
        Bin label for each density value
    point_counts : np.ndarray
        Number of points in each bin
    density_type : str
        'volumetric' or 'areal'
    output_dir : str or None
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved figure
    """
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Z bin (m)': bin_labels,
        'Local Density': local_densities
    })
    
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(data=df, x='Z bin (m)', y='Local Density', ax=ax, palette='Set2')
    
    ax.set_xlabel('Vertical Position (Z coordinate, m)', fontsize=12, fontweight='bold')
    unit = 'points/m³' if density_type == 'volumetric' else 'points/m²'
    ax.set_ylabel(f'Local Density ({unit}, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Vertical Density Distribution (Single Scan)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # Rotate x-axis labels if too many bins
    if len(np.unique(bin_labels)) > 8:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'vertical_density_violin.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nViolin plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_vertical_density_violin_aggregate(
    scan_heights, scan_densities, scan_ids, z_range,
    density_type='volumetric', output_dir=None,
    figsize=(12, 6), dpi=300,
    samples_per_scan=20000,
    weight_transform='log1p',     # 'none', 'log1p', or 'sqrt'
    clip_quantile=0.995,          # clip extreme weights
    density_norm='width',         # seaborn new name (replaces scale)
    random_seed=0,
    show_p25=True,
):
    """
    Density-weighted vertical violin plot (seaborn-version safe):
    - We emulate weights by resampling heights proportional to densities.
    - Violin width reflects density mass along height for each scan.

    samples_per_scan controls smoothness (bigger = smoother, slower).
    """
    import seaborn as sns

    rng = np.random.default_rng(random_seed)

    # -------- build resampled dataframe --------
    all_h = []
    all_sid = []

    n_scans = len(scan_heights)
    for i in range(n_scans):
        h = np.asarray(scan_heights[i], dtype=float)
        w = np.asarray(scan_densities[i], dtype=float)

        # scan id handling: allow passing a scalar id or per-point ids
        sid = scan_ids[i]
        if isinstance(sid, (list, tuple, np.ndarray)):
            sid_arr = np.asarray(sid)
            if sid_arr.size == 1:
                sid_val = sid_arr.item()
            else:
                # If per-point IDs were passed, we assume all same; otherwise fallback
                sid_val = sid_arr[0]
        else:
            sid_val = sid

        # filter bad values
        m = np.isfinite(h) & np.isfinite(w) & (w >= 0)
        h, w = h[m], w[m]
        if h.size == 0:
            continue

        # clip to plotting range (optional but makes KDE more stable)
        zmin, zmax = z_range
        m2 = (h >= zmin) & (h <= zmax)
        h, w = h[m2], w[m2]
        if h.size == 0:
            continue

        # clip extreme weights (prevents a few points dominating)
        if clip_quantile is not None and 0 < clip_quantile < 1:
            cap = np.quantile(w, clip_quantile)
            w = np.minimum(w, cap)

        # transform weights (recommended: log1p to tame heavy tails)
        if weight_transform == 'log1p':
            w = np.log1p(w)
        elif weight_transform == 'sqrt':
            w = np.sqrt(w)
        elif weight_transform == 'none':
            pass
        else:
            raise ValueError(f"Unknown weight_transform: {weight_transform}")

        # if all weights are zero after transform, fall back to uniform
        s = w.sum()
        if s <= 0:
            p = None
        else:
            p = w / s

        # resample heights according to weights
        n_samp = min(samples_per_scan, h.size) if p is None else samples_per_scan
        idx = rng.choice(h.size, size=n_samp, replace=True, p=p)
        h_rs = h[idx]

        all_h.append(h_rs)
        all_sid.append(np.full(h_rs.shape, sid_val))

    if len(all_h) == 0:
        raise ValueError("No valid data to plot after filtering.")

    df = pd.DataFrame({
        'Height (m)': np.concatenate(all_h),
        'Scan ID': np.concatenate(all_sid),
    })

    # stable scan ordering
    # if scan IDs are numeric-ish, this keeps numeric order
    try:
        ordered_ids = sorted(df['Scan ID'].unique(), key=lambda x: int(x))
    except Exception:
        ordered_ids = sorted(df['Scan ID'].unique(), key=lambda x: str(x))
    df['Scan ID'] = pd.Categorical(df['Scan ID'], categories=ordered_ids, ordered=True)

    # -------- style --------
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 13,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df, x='Scan ID', y='Height (m)',
        hue='Scan ID',               # same as x, but required for palette to work in newer seaborn
        palette='rainbow',
        legend=False,                # don't show redundant legend
        cut=0,
        inner='quartile',            # quartiles of *resampled* distribution (density-weighted)
        linewidth=0.5,
        density_norm=density_norm,   # replaces scale='width'
        ax=ax
    )

    ax.set_ylim(z_range)
    ax.set_xlabel('Scan ID', fontsize=15, fontweight='bold')
    ax.set_ylabel('Height (m)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1.2)
    ax.tick_params(labelsize=11, width=1.5, length=6)

    # Mark upright scans with asterisk
    upright_z_idx = [9, 17, 29, 34, 39]  # Each scan was inversed by default, except for these scans which were already upright, idx starts from 1
    
    # Modify x-tick labels to add asterisk for upright scans
    xtick_labels = [str(sid) if sid not in upright_z_idx else f"{sid}*" for sid in ordered_ids]
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    

    plt.tight_layout()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        outpath = out / 'vertical_density_violin_weighted_resample.png'
        fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {outpath}")

    return fig


