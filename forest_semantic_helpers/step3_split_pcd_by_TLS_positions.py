import numpy as np
import pandas as pd
from pathlib import Path
from tools.illustrate_spherical_projection import create_colored_cube_points
from forest_semantic_helpers.step0_read_pcd import observe_df

def align_cube_df_with_points(cube_df, points_df):
    """Align cube DataFrame columns to match points DataFrame structure."""
    points_df_cols = points_df.columns.tolist()
    cube_df_new = cube_df.copy()
    
    # Rename spatial coordinates to match points_df
    cube_df_new = cube_df_new.rename(columns={
        'x': 'X',
        'y': 'Y', 
        'z': 'Z'
    })
    
    # Add missing columns from points_df with default values
    for col in points_df_cols:
        if col not in cube_df_new.columns:
            if col == 'Intensity':
                cube_df_new[col] = 255  # High intensity for virtual points
            elif col == 'Classification':
                cube_df_new[col] = 9  # Classification code for virtual objects
            else:
                cube_df_new[col] = 0  # Default value
    
    # Reorder columns to match points_df
    cube_df_new = cube_df_new[points_df_cols + [col for col in cube_df_new.columns if col not in points_df_cols]]
    
    return cube_df_new


def add_virtual_cube_points(points_df, tls_positions):
    """Add virtual cube points to the points DataFrame based on TLS positions."""
    cube_dfs = []
    for i, tls_position in enumerate(tls_positions):
        print(f"TLS Position {i}: {tls_position}")
        scan_center = np.array([tls_position[0], tls_position[1], 2])
        cube_df = create_colored_cube_points(center=scan_center, size=0.5, samples_per_face=20)
        cube_df_new = align_cube_df_with_points(cube_df, points_df)
        
        # Assign the correct region_id to cube points
        if 'region_id' in points_df.columns:
            cube_df_new['region_id'] = i  # Assign cube to its corresponding region
        
        cube_dfs.append(cube_df_new)
    
    points_df_updated = pd.concat([points_df] + cube_dfs, ignore_index=True)
    return points_df_updated

def split_point_cloud_by_tls_scans(points_df, tls_positions_xy, generate_expanded_df=False, radius=20.0, verbose=True):
    """
    Creates overlapping regions where points can belong to multiple TLS scan regions.

    Parameters:
        points_df (pd.DataFrame): DataFrame with 'X', 'Y', 'Z' columns.
        tls_positions_xy (np.ndarray): Mx2 array of [X, Y] scan positions.
        radius (float): Radius of each circular region (in meters).

    Returns:
        pd.DataFrame: Expanded DataFrame where points can appear multiple times
                     with different region_id values if they belong to multiple regions.
    """
    region_dfs = []
    total_points_in_regions = 0
    
    for i, (cx, cy) in enumerate(tls_positions_xy):
        # Compute distances in XY plane
        distances = np.sqrt((points_df['X'] - cx) ** 2 + (points_df['Y'] - cy) ** 2)
        mask = distances <= radius
        
        # Create a copy of points within this region
        region_points = points_df[mask].copy()
        region_points['region_id'] = i  # Assign region ID
        if verbose:
            print("========== Region {}: Center ({:.2f}, {:.2f}), Radius {:.2f} ==========".format(i, cx, cy, radius))
            observe_df(region_points)
        
        region_dfs.append(region_points)
        region_count = len(region_points)
        total_points_in_regions += region_count
        print(f"Region {i}: {region_count} points within {radius} m of ({cx:.2f}, {cy:.2f})")
    
    points_df_dict = {f"region_{i:02d}": region_dfs[i] for i in range(len(tls_positions_xy))}
    
    
    if generate_expanded_df:
        points_df_expanded = pd.concat(region_dfs, ignore_index=True)
        points_df_final = add_virtual_cube_points(points_df_expanded, tls_positions_xy)
        points_df_dict["all_regions"] = points_df_final
        

    # Summary statistics
    original_count = len(points_df)
    
    print(f"Summary:")
    print(f"  Original points: {original_count}")
    print(f"  Number of regions: {len(tls_positions_xy)}")
    print(f"  Total points after region splitting: {total_points_in_regions}")

    
    return points_df_dict



def main():
    output_dir = Path("/home/fzhcis/Downloads/ForestSemantic/output")
    pts_path = output_dir / "Plot_1_rotated_with_sampled_ground.csv"
    # pts_path = Path("/home/fzhcis/Downloads/ForestSemantic/Plot_1.las")
    if not pts_path.exists():
        raise FileNotFoundError(f"File not found: {pts_path}")
    # points_df = read_pcd_file(pts_path, verbose=False)
    points_df = pd.read_csv(pts_path)
    print(f"Loaded {len(points_df):,} points from {pts_path.name}")
    tls_positions_path = output_dir / "tls_positions_xy.csv"
    region_radius = 13  # Radius for region overlap
    save_expanded_pts_df = False
    updated_output_path = output_dir / f"{pts_path.stem}_with_cubes_and_regions{region_radius:02d}.csv"
    

    if not tls_positions_path.exists():
        raise FileNotFoundError(f"File not found: {tls_positions_path}")
    tls_positions = pd.read_csv(tls_positions_path).values
    print(f"Detected {len(tls_positions)} TLS positions")


    # First split into regions
    points_by_tls_region = split_point_cloud_by_tls_scans(points_df=points_df, 
                                                                tls_positions_xy=tls_positions, 
                                                                radius=region_radius, 
                                                                generate_expanded_df=save_expanded_pts_df)

    for i in range(len(tls_positions)):
        export_csv_path = output_dir / f"region_{i:02d}_points.csv"
        print(f"Writing Region {i}: {len(points_by_tls_region[f'region_{i:02d}']):,} points to file {str(export_csv_path)}")
        points_by_tls_region[f'region_{i:02d}'].to_csv(
            export_csv_path, index=False
        )

    if save_expanded_pts_df:
        points_df_final = points_by_tls_region["all_regions"]
        points_df_final.to_csv(updated_output_path, index=False)
        print(f"Updated points with region IDs and cube data saved to: {updated_output_path}")
            


if __name__ == "__main__":
    main()
