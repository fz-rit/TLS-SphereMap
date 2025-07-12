import numpy as np
import pandas as pd
from pathlib import Path


def recenter_region_files(data_dir, tls_positions_df, output_dir):
    """Recenter individual region CSV files based on TLS positions."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    region_files = sorted(data_dir.glob("region_*_points.csv"))
    if not region_files:
        print(f"No region files found in {data_dir}")
        return []
    
    processed_files = []
    for region_file in region_files:
        try:
            # Extract region ID from filename
            region_id = int(region_file.stem.split('_')[1])
            region_df = pd.read_csv(region_file)
            
            # Get TLS position and recenter
            if region_id < len(tls_positions_df):
                tls_x = tls_positions_df.iloc[region_id]['X']
                tls_y = tls_positions_df.iloc[region_id]['Y']
                
                # Recenter coordinates
                recentered_df = region_df.copy()
                recentered_df['X'] -= tls_x
                recentered_df['Y'] -= tls_y
                
                # Save recentered file
                output_filename = f"region_{region_id:02d}_centered.csv"
                output_path = output_dir / output_filename
                recentered_df.to_csv(output_path, index=False)
                processed_files.append(output_path)
                
                # Report progress
                x_range = recentered_df['X'].max() - recentered_df['X'].min()
                y_range = recentered_df['Y'].max() - recentered_df['Y'].min()
                print(f"Region {region_id}: {len(region_df):,} points, range: {x_range:.1f}×{y_range:.1f}m")
            else:
                print(f"Warning: No TLS position for region {region_id}")
                
        except (ValueError, IndexError, Exception) as e:
            print(f"Error processing {region_file.name}: {e}")
    
    return processed_files


def main():
    """Main function to recenter region files."""
    data_dir = Path("/home/fzhcis/Downloads/ForestSemantic/output")
    tls_file = data_dir / "tls_positions_xy.csv"
    output_dir = data_dir / "recentered_regions"
    
    # Load TLS positions
    tls_positions_df = pd.read_csv(tls_file)
    print(f"Loaded {len(tls_positions_df)} TLS positions")
    
    # Process region files
    processed_files = recenter_region_files(data_dir, tls_positions_df, output_dir)
    print(f"Processed {len(processed_files)} region files → {output_dir}")


if __name__ == "__main__":
    main()
