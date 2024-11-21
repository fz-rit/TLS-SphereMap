import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from calc_extrinsic_matrix import extrinsic_mtx_face_to_origin

# Load the .ply file using Open3D
ply_file_path = Path("/home/fzhcis/mylab/data/point_cloud_segmentation/selected_data/cube_pointcloud.ply")

# def extend_point_cloud_fields:

pcd = o3d.io.read_point_cloud(str(ply_file_path))

# Convert to a numpy array
points = np.asarray(pcd.points)

# Define the LIDAR's extrinsic parameters
# Translation vector (LIDAR position in world coordinates)
lidar_position = (-3, -3, 3) # LiDAR position in world coordinates.
extrinsic_matrix = extrinsic_mtx_face_to_origin(lidar_position)


# Convert point cloud to homogeneous coordinates (add 1 as the fourth dimension)
homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

# Transform the points to the LIDAR's coordinate system
transformed_points = homogeneous_points @ extrinsic_matrix.T  # Apply extrinsic matrix
X, Y, Z = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]


# Calculate Range (distance from LIDAR to each point)
range_vals = np.sqrt(X**2 + Y**2 + Z**2)

# Calculate Azimuth (angle in XY plane from the positive X-axis)
azimuth_vals = np.degrees(np.arctan2(Y, X))
azimuth_vals = (azimuth_vals + 360) % 360  # Convert to range [0, 360]

# Calculate Zenith (angle from the positive Z-axis)
zenith_vals = np.degrees(np.arccos(Z / range_vals))

# Add Intensity Column with a constant value of 300
intensity_vals = np.full(len(points), 300, dtype=int)

# Add return number with a constant value of 1
return_num_vals = np.full(len(points), 1, dtype=int)

# Create a DataFrame to organize the data
angular_data = pd.DataFrame({
    'X': points[:, 0],
    'Y': points[:, 1],
    'Z': points[:, 2],
    'zenith': zenith_vals,
    'azimuth': azimuth_vals,
    'range1metres': range_vals,
    'Intensity': intensity_vals,
    'Return Number': return_num_vals,
})

# Save the DataFrame to a text file
output_file_path = ply_file_path.parent / f"cube_point_cloud_angular_space_{lidar_position[0]}_{lidar_position[1]}_{lidar_position[2]}.txt"
angular_data.to_csv(output_file_path, index=False, sep=',')

print(f"Extended Point cloud saved to {output_file_path}")
