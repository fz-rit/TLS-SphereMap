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
    # return Path(config["filename"]), Path(config["output_dir"])
    return config

def calc_normals_of_pt_cloud(filename, output_dir, range_min, range_max, visualize=False):
    """Process the point cloud by estimating normals and saving the result."""
    # Preprocess the point cloud
    df_filtered = preprocess_point_cloud(filename, range1metres_max=range_max, range1metres_min=range_min)

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
    print(df_filtered.head())
    print(f"Output dataframe shape: {df_filtered.shape}.")
    print(f'Point cloud with normals calculated!')

    # Save the result to a text file
    output_file_path = output_dir / f'{filename.stem}_filtered_normaled.txt'
    df_filtered.to_csv(output_file_path, sep=',', index=False)
    print(f'Point cloud with normals saved to {output_file_path}!')

    # Visualize to check the normals
    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def main():
    # Load the configuration file for input paths
    config_path = Path('./input_params/calc_pt_cloud_normal_inputs_amiri.json')
    config_dic = load_config(config_path)
    filename, output_dir = Path(config_dic["filename"]), Path(config_dic["output_dir"])
    range_min, range_max = config_dic["range1metres_min"], config_dic["range1metres_max"]
    visualize = config_dic["visualize"]

    # Process the point cloud
    calc_normals_of_pt_cloud(filename, output_dir, 
                             range_min=range_min, 
                             range_max=range_max,
                             visualize=visualize)

if __name__ == "__main__":
    main()
