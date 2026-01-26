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
        npz_path = output_path.with_suffix('.npz')
        json_path = output_path.with_suffix('.json')
        
        # Prepare arrays for NPZ (density vs range)
        arrays_to_save = {
            'bin_centers': data['bin_centers'],
            'densities_median': data['densities_median'],
            'densities_q25': data['densities_q25'],
            'densities_q75': data['densities_q75'],
            'point_counts': data['point_counts']
        }
        
        # Add per-scan data
        for i, (scan_dens, scan_centers) in enumerate(zip(data['scan_densities'], data['scan_bin_centers'])):
            arrays_to_save[f'scan_{i}_densities'] = scan_dens
            arrays_to_save[f'scan_{i}_bin_centers'] = scan_centers
        
        np.savez_compressed(npz_path, **arrays_to_save)
        
        # Save metadata to JSON
        metadata = {
            'mode': mode,
            'density_type': data.get('density_type', 'volumetric'),
            'n_scans': len(data['scan_densities'])
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDensity vs range results saved to:")
        print(f"  Data: {npz_path}")
        print(f"  Metadata: {json_path}")
        
        # Save vertical density distribution data
        if 'scan_heights' in data:
            vertical_npz = output_path.parent / (output_path.stem + '_vertical.npz')
            vertical_json = output_path.parent / (output_path.stem + '_vertical.json')
            
            # Prepare arrays
            vertical_arrays = {}
            for i, (heights, densities, ids) in enumerate(zip(
                data['scan_heights'], data['scan_densities_vertical'], data['scan_ids']
            )):
                vertical_arrays[f'scan_{i}_heights'] = heights
                vertical_arrays[f'scan_{i}_densities'] = densities
                vertical_arrays[f'scan_{i}_ids'] = np.array(ids)
            
            np.savez_compressed(vertical_npz, **vertical_arrays)
            
            # Save vertical metadata
            vertical_metadata = {
                'mode': mode,
                'density_type': data.get('density_type', 'volumetric'),
                'n_scans': len(data['scan_heights']),
                'z_range': list(data['z_range'])
            }
            with open(vertical_json, 'w') as f:
                json.dump(vertical_metadata, f, indent=2)
            
            print(f"\nVertical density results saved to:")
            print(f"  Data: {vertical_npz}")
            print(f"  Metadata: {vertical_json}")
    
    elif mode == 'single':
        npz_path = output_path.with_suffix('.npz')
        json_path = output_path.with_suffix('.json')
        
        arrays_to_save = {
            'bin_centers': data['bin_centers'],
            'densities_median': data['densities_median'],
            'densities_q25': data['densities_q25'],
            'densities_q75': data['densities_q75'],
            'point_counts': data['point_counts']
        }
        
        np.savez_compressed(npz_path, **arrays_to_save)
        
        metadata = {
            'mode': mode,
            'density_type': data.get('density_type', 'volumetric')
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  Data: {npz_path}")
        print(f"  Metadata: {json_path}")


def load_results(results_path):
    """
    Load intermediate computation results from files.
    
    Parameters:
    -----------
    results_path : str or Path
        Path to the .npz or .json file (either works)
        
    Returns:
    --------
    data : dict
        Dictionary containing loaded data
    metadata : dict
        Dictionary containing metadata
    """
    results_path = Path(results_path)
    npz_path = results_path.with_suffix('.npz')
    json_path = results_path.with_suffix('.json')
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    # Load arrays
    npz_data = np.load(npz_path)
    data = dict(npz_data)
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct per-scan data if aggregate mode
    if metadata['mode'] == 'aggregate':
        n_scans = metadata['n_scans']
        
        # Check if this is vertical density data (has heights) or density vs range data (has bin_centers)
        if 'scan_0_heights' in data:
            # Vertical density data
            pass  # Heights and IDs are already in the data dict
        elif 'scan_0_bin_centers' in data:
            # Density vs range data
            scan_densities = []
            scan_bin_centers = []
            
            for i in range(n_scans):
                scan_densities.append(data[f'scan_{i}_densities'])
                scan_bin_centers.append(data[f'scan_{i}_bin_centers'])
            
            data['scan_densities'] = scan_densities
            data['scan_bin_centers'] = scan_bin_centers
    
    print(f"\nLoaded results from: {npz_path}")
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


def plot_vertical_density_violin_aggregate(scan_heights, scan_densities, scan_ids, z_range,
                                          density_type='volumetric', output_dir=None,
                                          figsize=(12, 6), dpi=300):
    """
    Plot vertical density distribution with height on y-axis and scan ID on x-axis.
    Shows violin plots for each scan showing the distribution of heights.
    
    Parameters:
    -----------
    scan_heights : list of np.ndarray
        Heights (z-coordinates) for each scan
    scan_densities : list of np.ndarray
        Local densities corresponding to each height
    scan_ids : list of list
        Scan ID for each point
    z_range : tuple
        (z_min, z_max) for y-axis limits
    density_type : str
        'volumetric' or 'areal'
    output_dir : str or None
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        DPI for saved figure
    """
    import seaborn as sns
    
    # Set publication-quality style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    
    n_scans = len(scan_heights)
    
    # Prepare data for plotting
    all_heights = []
    all_scan_ids = []
    for scan_idx in range(n_scans):
        all_heights.extend(scan_heights[scan_idx])
        all_scan_ids.extend(scan_ids[scan_idx])
    
    df = pd.DataFrame({
        'Height (m)': all_heights,
        'Scan ID': all_scan_ids
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create vertical violin plot with scan ID on x-axis and height on y-axis
    sns.violinplot(data=df, x='Scan ID', y='Height (m)', ax=ax,
                   hue='Scan ID', palette='viridis', legend=False,
                   linewidth=2, inner='quartile',
                   cut=0, density_norm='width', saturation=0.8)
    
    # Compute and overlay median heights for each scan
    scan_medians = []
    for scan_idx in range(n_scans):
        if len(scan_heights[scan_idx]) > 0:
            median_height = np.median(scan_heights[scan_idx])
            scan_medians.append(median_height)
        else:
            scan_medians.append(np.nan)
    
    ax.scatter(range(n_scans), scan_medians,
              color='red', s=100, zorder=3, marker='D',
              label='Median Height', edgecolors='darkred', linewidths=1.5)
    
    # Styling
    ax.set_xlabel('Scan ID', fontsize=15, fontweight='bold')
    ax.set_ylabel('Height (m)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1.2)
    ax.tick_params(labelsize=12, width=1.5, length=6)
    
    # Set x-axis labels to show scan IDs
    ax.set_xticks(range(n_scans))
    # ax.set_xticklabels([f'Scan {i+1}' for i in range(n_scans)])
    
    # Rotate x-axis labels if too many scans
    if n_scans > 8:
        plt.xticks(rotation=45, ha='right')
    elif n_scans > 5:
        plt.xticks(rotation=30, ha='right')
    
    # Add legend
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.95,
                      edgecolor='black', fancybox=False, fontsize=13)
    legend.get_frame().set_linewidth(1.5)
    
    # Add statistics text
    total_points = sum(len(h) for h in scan_heights)
    z_min, z_max = z_range
    stats_text = (f'Total scans: {n_scans}\n'
                 f'Total points: {total_points:,}\n'
                 f'Height range: {z_min:.1f} - {z_max:.1f} m')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    plt.tight_layout()
    
    if output_dir:
        # Create output directory if it doesn't exist
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir_path / 'vertical_density_violin.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nVertical density violin plot saved to: {output_path}")
        print(f"Statistics: {n_scans} scans, {total_points:,} total points analyzed")
    else:
        plt.show()
    
    return fig

