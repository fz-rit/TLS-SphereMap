import open3d as o3d
import numpy as np
from matplotlib.colors import hsv_to_rgb

def normals_to_color(normals: np.ndarray) -> np.ndarray:
    # Step 2: Convert normals to spherical coordinates (azimuth and elevation)
    azimuth = np.arctan2(normals[:, 1], normals[:, 0])  # Angle in the XY plane
    elevation = np.arcsin(normals[:, 2])  # Angle relative to the Z-axis

    # Normalize azimuth and elevation to [0, 1] for HSV mapping
    hue = (azimuth + np.pi) / (2 * np.pi)  # Map azimuth from [-pi, pi] to [0, 1]
    value = (elevation + np.pi / 2) / np.pi  # Map elevation from [-pi/2, pi/2] to [0, 1]
    saturation = np.ones_like(hue)  # Set full saturation (1)

    # Combine into HSV format
    hsv_colors = np.stack([hue, saturation, value], axis=1)

    # Convert HSV to RGB
    rgb_colors = hsv_to_rgb(hsv_colors)

    # Step 3: Assign colors to the point cloud
    # Convert RGB to a format Open3D accepts
    colors = rgb_colors.astype(np.float64)
    return colors

if __name__ == "__main__":
    # Step 1: Load or create a point cloud
    # Create a simple sphere.
    point_cloud = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    point_cloud.compute_vertex_normals()  # Compute normals
    points = np.asarray(point_cloud.vertices)
    normals = np.asarray(point_cloud.vertex_normals)

    colors = normals_to_color(normals)

    point_cloud.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Step 4: Visualize the point cloud with normals as colors
    o3d.visualization.draw_geometries([point_cloud], window_name="Normals Visualized as Colors")
