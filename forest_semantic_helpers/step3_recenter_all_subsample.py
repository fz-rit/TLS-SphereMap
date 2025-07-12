import numpy as np
import pandas as pd
from pathlib import Path


def center_plot_by_tls(points_df, tls_positions_df, output_dir, plot_num):
    """Recenter individual region CSV files based on TLS positions."""

    for tls_pos_id in range(len(tls_positions_df)):
        print(f"Processing TLS Position {tls_pos_id + 1}/{len(tls_positions_df)}")
        tls_x = tls_positions_df.iloc[tls_pos_id]['X']
        tls_y = tls_positions_df.iloc[tls_pos_id]['Y']

        # Recenter coordinates
        recentered_df = points_df.copy()
        recentered_df['X'] -= tls_x
        recentered_df['Y'] -= tls_y
        recentered_df['Z'] -= recentered_df['Z'].mean()  # Center Z around mean height

        
        # Save recentered file
        output_filename = f"plot{plot_num}_centered_subsample_scan{tls_pos_id:02d}_{tls_x:.1f}_{tls_y:.1f}.csv"
        output_path = output_dir / output_filename
        recentered_df.to_csv(output_path, index=False)

        print(f"Recentered and subsampled {len(recentered_df):,} points to {output_path.name}")
        print(f"TLS Position {tls_pos_id}: ({tls_x:.1f}, {tls_y:.1f})")
        print(f"Spatial extent after recenter and subsample: \n \
              X: {recentered_df['X'].min():.1f} to {recentered_df['X'].max():.1f}, \
              Y: {recentered_df['Y'].min():.1f} to {recentered_df['Y'].max():.1f}, \
                Z: {recentered_df['Z'].min():.1f} to {recentered_df['Z'].max():.1f}")



def main():
    """Main function to recenter region files."""
    output_dir = Path(f"/home/fzhcis/Downloads/ForestSemantic/output/recenter_subsample")
    output_dir.mkdir(parents=True, exist_ok=True)
    for plot_num in [1, 3, 5]:
        print(f"======Processing Plot {plot_num}=======")
        data_dir = Path(f"/home/fzhcis/Downloads/ForestSemantic/output")
        tls_file = data_dir / f"Plot_{plot_num}_tls_positions_xy.csv"

        pts_path = data_dir / f"Plot_{plot_num}_rotated.csv"
        assert pts_path.exists(), f"File not found: {pts_path}"
        assert tls_file.exists(), f"File not found: {tls_file}"
        points_df = pd.read_csv(pts_path)
        tls_positions_df = pd.read_csv(tls_file)
        
        center_plot_by_tls(points_df, tls_positions_df, output_dir, plot_num)


if __name__ == "__main__":
    main()
