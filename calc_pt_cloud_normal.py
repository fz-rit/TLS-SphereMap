import open3d as o3d
import pandas as pd
import numpy as np
from pathlib import Path
from preprocess_point_cloud import preprocess_point_cloud

filename = Path(r'C:\Users\fzhcis\Documents\projects\from_RobC\for_Fei\data\palau_2024\ALRSET1\UMBCBL009_2024-03-28-02-47-26_ALRSET12_060180_000200.800_1830507489.txt')
output_dir = Path(r'C:\Users\fzhcis\Documents\mylab\tls_demo\outputs\1106\outputs')

df_filtered = preprocess_point_cloud(filename)

# Load your point cloud
# Create a PointCloud object and assign the points
points = df_filtered[['X', 'Y', 'Z']].to_numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)


# Estimate normals
# You can specify the radius for neighbors (in meters) or the number of neighbors
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optional: Orient normals consistently for better visualization
pcd.orient_normals_consistent_tangent_plane(k=10)

# # Visualize to check the normals
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Access the normals
normals = np.asarray(pcd.normals)
# print(normals)
# Add normals as new columns to the DataFrame
df_filtered[['nx', 'ny', 'nz']] = normals
print(df_filtered.head())
# Save the colorized point cloud to a text file

output_file_path = output_dir / f'{filename.stem}_filtered_normaled.txt'
df_filtered.to_csv(output_file_path, sep='\t', index=False)
print(f'Point cloud with normals saved to {output_file_path}!')
