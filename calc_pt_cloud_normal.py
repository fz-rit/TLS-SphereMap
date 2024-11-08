import open3d as o3d
import pandas as pd
import numpy as np
import json
from pathlib import Path
from preprocess_point_cloud import preprocess_point_cloud

def load_config(json_path):
    """Load configuration from a JSON file."""
    with open(json_path, 'r') as file:
        config = json.load(file)
    return Path(config["filename"]), Path(config["output_dir"])

def calc_normals_of_pt_cloud(filename, output_dir):
    """Process the point cloud by estimating normals and saving the result."""
    # Preprocess the point cloud
    df_filtered = preprocess_point_cloud(filename)

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

    # Save the result to a text file
    output_file_path = output_dir / f'{filename.stem}_filtered_normaled.txt'
    df_filtered.to_csv(output_file_path, sep='\t', index=False)
    print(f'Point cloud with normals saved to {output_file_path}!')

    # # Visualize to check the normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def main():
    # Load the configuration file for input paths
    config_path = Path('/home/felix/mylab/tls_point_segmentation/input_params/calc_pt_cloud_normal_inputs.json')
    filename, output_dir = load_config(config_path)

    # Process the point cloud
    calc_normals_of_pt_cloud(filename, output_dir)

if __name__ == "__main__":
    main()
