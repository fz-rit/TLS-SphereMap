import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
ZENITH_RES = 0.5
AZIMUTH_RES = 0.5
GROUP_SIZE = 10

def spherical_projection(df, zenith_res=ZENITH_RES, azimuth_res=AZIMUTH_RES):
    """
    Converts a point cloud with azimuth and zenith info into a 3xH×W RGB spherical image.
    
    Parameters:
        df: pandas DataFrame with columns x, y, z, zenith_deg, azimuth_deg, r, g, b
        zenith_res: resolution of zenith in degrees (default 0.25)
        azimuth_res: resolution of azimuth in degrees (default 0.25)
        
    Returns:
        img: np.ndarray of shape (3, H, W) representing RGB spherical projection
    """
    # Image dimensions
    H = int(135 / zenith_res) + 1  # rows: 0 to 135 inclusive
    W = int(360 / azimuth_res)     # cols: 0 to <360
    img = np.zeros((3, H, W), dtype=np.float32)

    # Compute row/col indices
    zenith_idx = (df['zenith_deg'] / zenith_res).astype(int)
    azimuth_idx = (df['azimuth_deg'] / azimuth_res).astype(int) % W  # wrap around 360

    # Clip to bounds just in case
    zenith_idx = np.clip(zenith_idx, 0, H - 1)
    azimuth_idx = np.clip(azimuth_idx, 0, W - 1)

    # Fill image
    for c, color in enumerate(['r', 'g', 'b']):
        img[c, zenith_idx, azimuth_idx] = df[color].values

    return img


def create_colored_cube(center=[0, 0, 0], size=1.0):
    s = size / 2.0
    cx, cy, cz = center

    # 8 vertices
    vertices = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ])

    # 12 triangles (2 per face)
    triangles = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [3, 0, 4], [3, 4, 7],  # left
    ]

    # Assign one color per vertex (not face), e.g., repeat colors for visible variety
    vertex_colors = np.array([
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [1, 1, 0],  # yellow
        [1, 0, 1],  # magenta
        [0, 1, 1],  # cyan
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.0],
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


def generate_lidar_ball(radius=20.0, zenith_range = (0, 135), res_deg=AZIMUTH_RES):
    """
    Generate a dense LiDAR-like point cloud with one point per angular bin.
    Output columns: x, y, z, azimuth_deg, zenith_deg, r, g, b
    Group color every 5° and add distinguishable lat/lon lines.
    """
    azimuths_deg = np.arange(0, 360, res_deg)
    zeniths_deg = np.arange(zenith_range[0], zenith_range[1] + res_deg, res_deg)

    azimuth_grid, zenith_grid = np.meshgrid(azimuths_deg, zeniths_deg)
    azimuth_flat = azimuth_grid.flatten()
    zenith_flat = zenith_grid.flatten()

    azimuth_rad = np.deg2rad(azimuth_flat)
    zenith_rad = np.deg2rad(zenith_flat)

    # Spherical to Cartesian
    x = radius * np.sin(zenith_rad) * np.cos(azimuth_rad)
    y = radius * np.sin(zenith_rad) * np.sin(azimuth_rad)
    z = radius * np.cos(zenith_rad)

    # Group by GROUP_SIZE° bins for coloring
    az_bin = (azimuth_flat // GROUP_SIZE).astype(int)
    ze_bin = (zenith_flat // GROUP_SIZE).astype(int)

    # Normalize to [0, 1] for HSV
    hue = ze_bin / ze_bin.max()           # Hue by zenith bin
    sat = az_bin / az_bin.max()           # Saturation by azimuth bin
    hsv = np.stack((hue, sat, np.ones_like(hue)), axis=1)
    rgb = hsv_to_rgb(hsv)

    # Highlight latitude lines
    lat_lines = np.arange(zenith_range[0]+GROUP_SIZE, zenith_range[1]+GROUP_SIZE, GROUP_SIZE)
    lat_mask = np.isclose(zenith_flat[:, None], lat_lines, atol=0.2).any(axis=1)

    # Highlight longitude lines
    lon_lines = np.arange(0, 360, GROUP_SIZE)
    lon_mask = np.isclose(azimuth_flat[:, None], lon_lines, atol=0.2).any(axis=1)

    # Combine masks and assign unique colors
    rgb[lat_mask] = np.array([1.0, 0.0, 0.0])  # Red for latitude lines
    rgb[lon_mask] = np.array([0.0, 0.0, 1.0])  # Blue for longitude lines

    # Pack into DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'azimuth_deg': azimuth_flat,
        'zenith_deg': zenith_flat,
        'r': rgb[:, 0],
        'g': rgb[:, 1],
        'b': rgb[:, 2],
    })

    return df



def spherical_projection_with_density(df, zenith_range = (0, 135), zenith_res=ZENITH_RES, azimuth_res=AZIMUTH_RES):
    """
    Efficiently compute RGB spherical projection and density map using groupby.
    
    Returns:
        rgb_img: (3, H, W) float32 image (RGB)
        density_map: (H, W) int32 image (point counts)
    """
    zenith_range_abs = zenith_range[1] - zenith_range[0]
    H = int(zenith_range_abs / zenith_res) + 1
    W = int(360 / azimuth_res)

    # Compute pixel indices
    zenith_deg_shifted = df['zenith_deg'] - df['zenith_deg'].min()
    print("Unique zenith degrees:", df['zenith_deg'].nunique())
    print("Zenith degrees min:", df['zenith_deg'].min())
    print("Zenith degrees max:", df['zenith_deg'].max())

    df['zenith_idx'] = (zenith_deg_shifted / zenith_res).astype(int).clip(0, H - 1)
    df['azimuth_idx'] = (df['azimuth_deg'] / azimuth_res).astype(int) % W

    # Group by pixel indices
    grouped = df.groupby(['zenith_idx', 'azimuth_idx'])

    # Compute mean RGB and density
    mean_rgb = grouped[['r', 'g', 'b']].mean().reset_index()
    density = grouped.size().reset_index(name='count')

    # Initialize output
    rgb_img = np.zeros((3, H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.int32)

    # Assign values using .loc
    for _, row in mean_rgb.iterrows():
        z, a = int(row['zenith_idx']), int(row['azimuth_idx'])
        rgb_img[0, z, a] = row['r']
        rgb_img[1, z, a] = row['g']
        rgb_img[2, z, a] = row['b']

    for _, row in density.iterrows():
        z, a = int(row['zenith_idx']), int(row['azimuth_idx'])
        density_map[z, a] = row['count']

    return rgb_img, density_map

def plot_and_save_results(rgb_img, density_map, zenith_range=(0, 135), output_prefix='spherical', res_deg=AZIMUTH_RES):
    # Convert and save RGB image
    rgb_display = np.transpose(rgb_img, (1, 2, 0))  # (H, W, 3)
    rgb_uint8 = (rgb_display * 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(f'outputs/{output_prefix}_rgb_image.png')

    # Define discrete bins and colors
    flat_density = density_map.flatten()
    max_val = np.max(flat_density)
    last_bin = max(4, int(max_val) + 1)
    bins = [0, 1, 2, 3, last_bin]
    labels = ['0', '1', '2', '>2']

    colors = [
                '#ffffcc',  # light yellow
                '#a1dab4',  # greenish-teal
                '#41b6c4',  # medium cyan
                '#225ea8',  # dark blue
            ]
    # Assign bin labels to density map
    digitized = np.digitize(density_map, bins, right=False) - 1
    density_colored = np.zeros((density_map.shape[0], density_map.shape[1], 3), dtype=np.uint8)

    for i, hex_color in enumerate(colors):
        mask = digitized == i
        rgb = tuple(int(hex_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        density_colored[mask] = rgb

    Image.fromarray(density_colored).save(f'outputs/{output_prefix}_density_map.png')

    # Create axis ticks for azimuth and zenith
    height, width = density_map.shape
    azimuth_ticks = np.linspace(0, 360, num=9)  # every 45°
    zenith_ticks = np.linspace(zenith_range[0], zenith_range[1], num=6)   # every ~27°

    azimuth_pos = (azimuth_ticks / res_deg).astype(int)
    zenith_pos = (zenith_ticks / res_deg).astype(int)

    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))

    # --- RGB Image ---
    axs[0].imshow(rgb_display)
    axs[0].set_title("Spherical RGB Projection")
    axs[0].set_xlabel("Azimuth Angle (°)")
    axs[0].set_ylabel("Zenith Angle (°)")
    axs[0].set_xticks(azimuth_pos)
    axs[0].set_xticklabels([str(int(t)) for t in azimuth_ticks])
    axs[0].set_yticks(zenith_pos)
    axs[0].set_yticklabels([str(int(t)) for t in zenith_ticks])

    # --- Density Map ---
    axs[1].imshow(density_colored)
    axs[1].set_title("Point Density Map (Discrete Colors)")
    axs[1].set_xlabel("Azimuth Angle (°)")
    axs[1].set_ylabel("Zenith Angle (°)")
    axs[1].set_xticks(azimuth_pos)
    axs[1].set_xticklabels([str(int(t)) for t in azimuth_ticks])
    axs[1].set_yticks(zenith_pos)
    axs[1].set_yticklabels([str(int(t)) for t in zenith_ticks])
    axs[1].legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], loc='upper right')

    # --- Pie Chart ---
    hist, _ = np.histogram(flat_density, bins=bins)
    axs[2].pie(hist, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    axs[2].set_title("Pixel-wise Point Density Distribution")

    plt.tight_layout()
    plt.savefig(f'outputs/{output_prefix}_combined_plot.png')
    print(f"Saved:outputs/{output_prefix}_rgb_image.png, {output_prefix}_density_map.png, {output_prefix}_combined_plot.png")

def create_colored_cube_points(center=[0, 0, 0], size=1.0, samples_per_face=100):
    """
    Generate uniformly sampled colored points from each cube face.
    Returns a DataFrame with x, y, z, r, g, b, azimuth_deg, zenith_deg.
    """
    cx, cy, cz = center
    s = size / 2.0
    lin = np.linspace(-s, s, samples_per_face)
    X, Y = np.meshgrid(lin, lin)
    X = X.flatten()
    Y = Y.flatten()

    faces = [
        {'coord': 'z', 'val': -s, 'axes': ('x', 'y'), 'color': [1, 0, 0]},  # Bottom
        {'coord': 'z', 'val': +s, 'axes': ('x', 'y'), 'color': [0, 1, 0]},  # Top
        {'coord': 'y', 'val': -s, 'axes': ('x', 'z'), 'color': [0, 0, 1]},  # Front
        {'coord': 'y', 'val': +s, 'axes': ('x', 'z'), 'color': [1, 1, 0]},  # Back
        {'coord': 'x', 'val': -s, 'axes': ('y', 'z'), 'color': [1, 0, 1]},  # Left
        {'coord': 'x', 'val': +s, 'axes': ('y', 'z'), 'color': [0, 1, 1]},  # Right
    ]

    all_points = []

    for face in faces:
        coord = face['coord']
        val = face['val']
        a1, a2 = face['axes']
        color = face['color']

        pts = {a1: X, a2: Y, coord: np.full_like(X, val)}
        pts['x'] = pts.get('x', np.zeros_like(X)) + cx
        pts['y'] = pts.get('y', np.zeros_like(X)) + cy
        pts['z'] = pts.get('z', np.zeros_like(X)) + cz

        # Compute spherical angles
        x = pts['x']
        y = pts['y']
        z = pts['z']
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.degrees(np.arctan2(y, x)) % 360
        zenith = np.degrees(np.arccos(z / r))

        df_face = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'r': np.full_like(x, color[0]),
            'g': np.full_like(x, color[1]),
            'b': np.full_like(x, color[2]),
            'azimuth_deg': azimuth,
            'zenith_deg': zenith,
        })

        all_points.append(df_face)

    cube_df = pd.concat(all_points, ignore_index=True)
    return cube_df



def visualize_point_cloud(df, add_cube = False):
    """
    Visualize the point cloud using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
    pcd.colors = o3d.utility.Vector3dVector(df[['r', 'g', 'b']].values)

    # Visualize
    if add_cube:
        cube_mesh = create_colored_cube(center=[0, 0, 0], size=1.0)
        o3d.visualization.draw_geometries([pcd, cube_mesh], window_name="Point Cloud with Cube")
    else:
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")


def visualize_and_save_point_cloud(df_in, zenith_range=(0, 135), visualize=True, save_dir=None, output_prefix='spherical'):
    if visualize:
        visualize_point_cloud(df_in)
    
    if save_dir is None:
        save_dir = Path('outputs')
    output_path = save_dir / f'{output_prefix}_sphere.csv'
    df_in.to_csv(output_path, index=False)
    print(f"Saved combined point cloud to {output_path}")
    # Perform spherical projection
    rgb_img, density_map = spherical_projection_with_density(df_in, zenith_range=zenith_range)
    plot_and_save_results(rgb_img, density_map, 
                          zenith_range=zenith_range, 
                          output_prefix=output_prefix, 
                          res_deg=AZIMUTH_RES)


def main():
    # Generate a dense LiDAR-like point cloud
    visualize = True
    df_ball = generate_lidar_ball(radius=10.0)
    visualize_and_save_point_cloud(df_ball, visualize=visualize, output_prefix="ball")

    df_cube = create_colored_cube_points(center=[3, 3, 3], size=5.0, samples_per_face=50)
    df_combined = pd.concat([df_ball, df_cube], ignore_index=True)
    visualize_and_save_point_cloud(df_combined, visualize=visualize, output_prefix="ball_cube")
    

if __name__ == "__main__":
    main()
    print("Processing complete.")