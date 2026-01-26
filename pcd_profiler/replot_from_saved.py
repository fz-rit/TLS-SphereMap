"""
Replot results from saved intermediate data files.

This script allows you to modify plots without recomputing density values.
Simply load the saved .npz/.json files and regenerate plots with different
styling, figure sizes, or other visualization parameters.

Usage:
    python replot_from_saved.py <results_path> [options]
    
Example:
    python replot_from_saved.py output/density_results.npz --figsize 10 8 --dpi 150
"""

import argparse
import numpy as np
from pathlib import Path

from density_io_plot import (load_results, plot_density_vs_range,
                             plot_aggregate_density_vs_range,
                             plot_vertical_density_violin_aggregate)


def main():
    parser = argparse.ArgumentParser(
        description='Replot density analysis from saved results'
    )
    parser.add_argument('results_path', type=str,
                       help='Path to saved results (.npz or .json file)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for new plots (default: creates _replot directory)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 6],
                       help='Figure size (width height) in inches')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure')
    parser.add_argument('--plot-type', type=str, 
                       choices=['density', 'vertical', 'both'], default='both',
                       help='Which plot to generate')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    
    # Load results
    print("="*60)
    print("Loading saved results...")
    print("="*60)
    data, metadata = load_results(results_path)
    
    mode = metadata['mode']
    density_type = metadata.get('density_type', 'volumetric')
    figsize = tuple(args.figsize)
    
    # Generate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / (results_path.stem.replace('_results', '') + '_replot')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMode: {mode.upper()}")
    print(f"Density type: {density_type}")
    print(f"Output directory: {output_dir}")
    
    # Plot based on mode and type
    if mode == 'aggregate' and args.plot_type in ['density', 'both']:
        print("\n" + "="*60)
        print("Generating density vs range plot...")
        print("="*60)
        
        plot_aggregate_density_vs_range(
            data['bin_centers'],
            data['densities_median'],
            data['densities_q25'],
            data['densities_q75'],
            data['point_counts'],
            data['scan_densities'],
            data['scan_bin_centers'],
            density_type=density_type,
            output_dir=str(output_dir),
            figsize=figsize,
            dpi=args.dpi
        )
    
    elif mode == 'single' and args.plot_type in ['density', 'both']:
        print("\n" + "="*60)
        print("Generating density vs range plot...")
        print("="*60)
        
        plot_density_vs_range(
            data['bin_centers'],
            data['densities_median'],
            data['densities_q25'],
            data['densities_q75'],
            data['point_counts'],
            density_type=density_type,
            output_dir=str(output_dir),
            figsize=figsize,
            dpi=args.dpi
        )
    
    # Try to load and plot vertical density if requested
    if args.plot_type in ['vertical', 'both']:
        vertical_path = results_path.parent / (results_path.stem + '_vertical')
        
        try:
            print("\n" + "="*60)
            print("Loading vertical density data...")
            print("="*60)
            vertical_data, vertical_metadata = load_results(vertical_path)
            
            # Reconstruct scan data
            n_scans = vertical_metadata['n_scans']
            scan_heights = []
            scan_densities = []
            scan_ids = []
            
            for i in range(n_scans):
                scan_heights.append(vertical_data[f'scan_{i}_heights'])
                scan_densities.append(vertical_data[f'scan_{i}_densities'])
                scan_ids.append(vertical_data[f'scan_{i}_ids'].tolist())
            
            z_range = tuple(vertical_metadata['z_range'])
            
            print("\n" + "="*60)
            print("Generating vertical density violin plot...")
            print("="*60)
            
            plot_vertical_density_violin_aggregate(
                scan_heights, scan_densities, scan_ids, z_range,
                density_type=density_type,
                output_dir=str(output_dir),
                figsize=figsize,
                dpi=args.dpi
            )
            
        except FileNotFoundError:
            print(f"\nVertical density data not found at {vertical_path}")
            print("Skipping vertical density plot.")
    
    print("\n" + "="*60)
    print("Replotting complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
