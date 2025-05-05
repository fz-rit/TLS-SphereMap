import open3d as o3d
import numpy as np
from matplotlib.colors import hsv_to_rgb
from pandas import DataFrame

def normals_to_color(normals: np.ndarray) -> np.ndarray:
    normals = np.round(normals, decimals=5)
    # Step 2: Convert normals to spherical coordinates (azimuth and elevation)
    if np.isnan(normals).any():
        print("Normals contain NaN values")
        raise ValueError("Normals should not contain NaN values")
    # Ensure normals are within the range [-1, 1]
    if not np.all((normals >= -1) & (normals <= 1)):
        print(f"Normals range per column: \
              \nnx min={normals[:, 0].min()}, \
              \nnx max={normals[:, 0].max()}, \
              \nny min={normals[:, 1].min()}, \
              \nny max={normals[:, 1].max()}, \
              \nnz min={normals[:, 2].min()}, \
              \nnz max={normals[:, 2].max()}")
        raise ValueError("Normals should be in the range [-1, 1]")
    
    # Flip nz if most of them are negative
    if np.mean(normals[:, 2]) < 0:
        normals[:, 2] = -normals[:, 2]
    azimuth = np.arctan2(normals[:, 1], normals[:, 0])  # Angle in the XY plane
    elevation = np.arcsin(normals[:, 2])  # Angle relative to the Z-axis

    # Normalize azimuth and elevation to [0, 1] for HSV mapping
    hue = (azimuth + np.pi) / (2 * np.pi)  # Map azimuth from [-pi, pi] to [0, 1]
    value = (elevation + np.pi / 2) / np.pi  # Map elevation from [-pi/2, pi/2] to [0, 1]
    saturation = np.ones_like(hue).astype(float) / 2  # 0.5 produces smoother, more balanced histograms than 1.0

    # Combine into HSV format
    hsv_colors = np.stack([hue, saturation, value], axis=1)

    # Convert HSV to RGB
    rgb_colors = hsv_to_rgb(hsv_colors)

    # Convert RGB to a format Open3D accepts
    colors = rgb_colors.astype(np.float64)
    return colors

def attach_normal_color_to_df(df: DataFrame) -> DataFrame:
    """
    Attach normal vector colors to a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing normal vectors with columns 'nx', 'ny', 'nz'.
    normals (np.ndarray): Array of normal vectors.

    Returns:
    pd.DataFrame: DataFrame with additional columns 'n_r', 'n_g', 'n_b' representing the RGB colors.
    """
    normals = df[['nx', 'ny', 'nz']].values
    colors = normals_to_color(normals)
    df['n_r'] = colors[:, 0]
    df['n_g'] = colors[:, 1]
    df['n_b'] = colors[:, 2]
    return df

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
