"""
Contributor: fzhcis@rit.edu
Version: 1.0
Last Updated: 11/19/2024
Description:
This script calculates the normals of a point cloud and saves the result to a text file.
"""

import open3d as o3d
import pandas as pd
import numpy as np
import json
from pathlib import Path
from preprocess_point_cloud import preprocess_point_cloud
from config_loader import CONFIG

# Ignore warnings
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning


def calc_normals_of_pt_cloud(input_path: Path, 
                             output_dir: Path, 
                             range_min:float, 
                             range_max:float, 
                             clean_pc:bool, 
                             visualize:bool=False,
                             flip_mangrove:bool=True):
    """Process the point cloud by estimating normals and saving the result."""
    # Filter the point cloud by range values
    df_filtered = preprocess_point_cloud(input_path, 
                                         range1metres_max=range_max, 
                                         range1metres_min=range_min,
                                         clean_pc=clean_pc,
                                         flip_mangrove=flip_mangrove)

    # Load point cloud data and create PointCloud object
    points = df_filtered[['X', 'Y', 'Z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Add normals to the DataFrame
    normals = np.asarray(pcd.normals)
    df_filtered[['nx', 'ny', 'nz']] = normals
    df_filtered = df_filtered.astype({ # the default type for float is 'float64', which consumes more space than necessary.
                                    'X': 'float32',
                                    'Y': 'float32',
                                    'Z': 'float32',
                                    'Intensity': 'float32',
                                    'Return Number': 'uint8',
                                    'azimuth': 'float32',
                                    'zenith': 'float32',
                                    'elevation': 'float32',
                                    'range1metres': 'float32',
                                    'nx': 'float32',
                                    'ny': 'float32',
                                    'nz': 'float32'
                                })
    print(df_filtered.head())
    print(f"Output dataframe shape: {df_filtered.shape}.")
    print(f'Point cloud with normals calculated!')

    # Save the result to a text file
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f'Output directory does not exist! Now created at {output_dir}!')
    output_file_path = output_dir / f'{input_path.stem}_filtered_normaled.txt'
    df_filtered.to_csv(output_file_path, sep=',', index=False, float_format='%.5f')
    print(f'Point cloud with normals saved to {output_file_path}!')

    # Visualize to check the normals
    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def main():
    # Load the configuration file for input paths
    params = CONFIG["calc_pt_cloud_normal"]
    global_params = CONFIG["global"]
    output_dir = Path(global_params["output_dir"])
    input_path = Path(f"{global_params['input_base_dir']}/{global_params['input_folder']}/{global_params['input_file_stem']}.txt")
    range_min, range_max = params["range1metres_min"], params["range1metres_max"]
    clean_pc = params["clean_pc"]
    visualize = params["visualize"]
    flip_mangrove = params["flip_mangrove"]

    # Process the point cloud
    calc_normals_of_pt_cloud(input_path, output_dir, 
                             range_min=range_min, 
                             range_max=range_max,
                             clean_pc=clean_pc,
                             visualize=visualize,
                             flip_mangrove=flip_mangrove)

if __name__ == "__main__":
    main()
