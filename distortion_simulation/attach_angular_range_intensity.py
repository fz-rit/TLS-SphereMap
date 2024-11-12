import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path

# Load the .ply file using Open3D
ply_file_path = Path(r"C:\Users\fzhcis\Documents\mylab\tls_demo\demo_files\cube_pointcloud.ply")

def extend_point_cloud_fields:

pcd = o3d.io.read_point_cloud(str(ply_file_path))

# Convert to a numpy array
points = np.asarray(pcd.points)

# Assuming the LIDAR is 3 meters along the negative Z-axis
lidar_position = np.array([-4, -4, -4])

# Adjust points relative to the LIDAR position
adjusted_points = points - lidar_position  # Shift each point by the LIDAR position
X, Y, Z = adjusted_points[:, 0], adjusted_points[:, 1], adjusted_points[:, 2]

# Calculate Range (distance from LIDAR to each point)
range_vals = np.sqrt(X**2 + Y**2 + Z**2)

# Calculate Azimuth (angle in XY plane from the positive X-axis)
azimuth_vals = np.degrees(np.arctan2(Y, X))

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
output_file_path = ply_file_path.parent / "point_cloud_angular_space.txt"
angular_data.to_csv(output_file_path, index=False, sep=',', header=False)

print(f"Extended Point cloud saved to {output_file_path}")
