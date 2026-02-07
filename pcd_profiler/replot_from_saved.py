"""
Replot vertical density violin plots from saved CSV data.

This script allows you to regenerate vertical density plots without recomputing
density values. Simply load the saved vertical_density.csv and regenerate plots
with different styling, figure sizes, or visualization parameters.

Usage:
    python replot_from_saved.py [options]
    
Example:
    python replot_from_saved.py --data-dir /path/to/results --figsize 14 8 --dpi 150 --samples 30000
"""

import argparse
import pandas as pd
from pathlib import Path

from density_io_plot import plot_vertical_density_violin_aggregate


def main():
    parser = argparse.ArgumentParser(
        description='Replot vertical density violin plots from saved CSV data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing vertical_density.csv and vertical_density_metadata.json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for new plots (default: same as data-dir with _replot suffix)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 6],
                       help='Figure size (width height) in inches')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure')
    parser.add_argument('--samples', type=int, default=20000,
                       help='Number of samples per scan for violin plot smoothness')
    parser.add_argument('--weight-transform', type=str, default='log1p',
                       choices=['none', 'log1p', 'sqrt'],
                       help='Transform applied to density weights')
    parser.add_argument('--clip-quantile', type=float, default=0.995,
                       help='Clip extreme weights at this quantile (0-1, or None to disable)')
    parser.add_argument('--show-p25', action='store_true', default=True,
                       help='Show P25 markers on plot')
    parser.add_argument('--no-p25', dest='show_p25', action='store_false',
                       help='Hide P25 markers on plot')
    parser.add_argument('--random-seed', type=int, default=0,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load vertical density data
    print("="*60)
    print("Loading vertical density data...")
    print("="*60)
    
    csv_path = data_dir / 'vertical_density.csv'
    json_path = data_dir / 'vertical_density_metadata.json'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    # Load CSV and metadata
    df = pd.read_csv(csv_path)
    import json
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"  Loaded {len(df)} data points")
    print(f"  Number of scans: {metadata['n_scans']}")
    print(f"  Density type: {metadata['density_type']}")
    
    # Reconstruct scan data
    scan_heights = []
    scan_densities = []
    scan_ids = []
    
    for scan_id in sorted(df['scan_id'].unique()):
        scan_data = df[df['scan_id'] == scan_id]
        scan_heights.append(scan_data['height'].values)
        scan_densities.append(scan_data['density'].values)
        scan_ids.append(scan_id)
    
    z_range = tuple(metadata['z_range'])
    density_type = metadata.get('density_type', 'volumetric')
    
    # Generate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'replot'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVisualization parameters:")
    print(f"  Figure size: {args.figsize[0]} x {args.figsize[1]} inches")
    print(f"  DPI: {args.dpi}")
    print(f"  Samples per scan: {args.samples}")
    print(f"  Weight transform: {args.weight_transform}")
    print(f"  Clip quantile: {args.clip_quantile}")
    print(f"  Show P25: {args.show_p25}")
    print(f"  Output directory: {output_dir}")
    
    # Generate plot
    print("\n" + "="*60)
    print("Generating vertical density violin plot...")
    print("="*60)
    
    clip_val = args.clip_quantile if args.clip_quantile > 0 else None
    
    plot_vertical_density_violin_aggregate(
        scan_heights, scan_densities, scan_ids, z_range,
        density_type=density_type,
        output_dir=str(output_dir),
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        samples_per_scan=args.samples,
        weight_transform=args.weight_transform,
        clip_quantile=clip_val,
        random_seed=args.random_seed,
        show_p25=args.show_p25
    )
    
    print("\n" + "="*60)
    print("Replotting complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
