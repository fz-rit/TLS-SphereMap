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
from tools.preprocess_point_cloud import read_and_clean_pcd
from tools.config_loader import CONFIG
from tools.pcd_utils import create_dir_if_not_exists

# Ignore warnings
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning


def calc_normals_of_pt_cloud(input_path: Path, 
                             output_dir: Path, 
                            cut_percent:float,
                             clean_pc:bool, 
                             visualize:bool=False,
                             flip_mangrove:bool=True):
    """Process the point cloud by estimating normals and saving the result."""
    # Filter the point cloud by range values
    df_filtered = read_and_clean_pcd(input_path, 
                                         cut_percent=cut_percent,
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
                                    'rangemeter': 'float32',
                                    'nx': 'float32',
                                    'ny': 'float32',
                                    'nz': 'float32'
                                })
    print(df_filtered.head())
    print(f"Output dataframe shape: {df_filtered.shape}.")
    print(f'Point cloud with normals calculated!')

    save_dir = output_dir / 'pcd'
    create_dir_if_not_exists(save_dir, ask_user=False)
    output_file_path = save_dir / f'{input_path.stem}_filtered_normaled.txt'
    df_filtered.to_csv(output_file_path, sep=',', index=False, float_format='%.5f')
    print(f'Point cloud with normals saved to {output_file_path}!')

    # Visualize to check the normals
    if visualize:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def main():
    # Load the configuration file for input paths
    params = CONFIG["calc_pt_cloud_normal"]
    global_params = CONFIG["global"]
    output_dir_ls = global_params["output_dir_ls"]
    input_path_ls = global_params["input_path_ls"]
    cut_percent = params["cut_percent"]
    clean_pc = params["clean_pc"]
    visualize = params["visualize"]
    flip_mangrove = params["flip_mangrove"]

    # Process the point cloud
    for input_path, output_dir in zip(input_path_ls, output_dir_ls):
        print(f'#######Processing {input_path}...########')
        
        calc_normals_of_pt_cloud(input_path, output_dir, 
                                cut_percent = cut_percent,
                                clean_pc = clean_pc,
                                visualize = visualize,
                                flip_mangrove = flip_mangrove)

if __name__ == "__main__":
    main()
