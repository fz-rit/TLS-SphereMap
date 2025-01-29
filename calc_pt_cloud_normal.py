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

# Ignore warnings
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

def load_config(json_path):
    """Load configuration from a JSON file."""
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config

def calc_normals_of_pt_cloud(filename, output_dir, range_min, range_max, clean_pc, visualize=False):
    """Process the point cloud by estimating normals and saving the result."""
    # Filter the point cloud by range values
    df_filtered = preprocess_point_cloud(filename, 
                                         range1metres_max=range_max, 
                                         range1metres_min=range_min,
                                         clean_pc=clean_pc)

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
    output_file_path = output_dir / f'{filename.stem}_filtered_normaled.txt'
    df_filtered.to_csv(output_file_path, sep=',', index=False, float_format='%.5f')
    print(f'Point cloud with normals saved to {output_file_path}!')

    # Visualize to check the normals
    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def main():
    # Load the configuration file for input paths
    config_path = Path('./input_params/calc_pt_cloud_normal_inputs_zmachine_harvard.json')
    config_dic = load_config(config_path)
    filename, output_dir = Path(config_dic["filename"]), Path(config_dic["output_dir"])
    range_min, range_max = config_dic["range1metres_min"], config_dic["range1metres_max"]
    clean_pc = config_dic["clean_pc"]
    visualize = config_dic["visualize"]

    # Process the point cloud
    calc_normals_of_pt_cloud(filename, output_dir, 
                             range_min=range_min, 
                             range_max=range_max,
                             clean_pc=clean_pc,
                             visualize=visualize)

if __name__ == "__main__":
    main()
