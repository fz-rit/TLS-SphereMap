"""
Run this file with `python run_3d_to_2d_pipeline.py` or `python -m processing.spherical_back_projection`

"""

import numpy as np
from tools.illustrate_spherical_projection import (generate_lidar_ball, 
                                             visualize_and_save_point_cloud, 
                                             create_colored_cube_points)
from PIL import Image
import pandas as pd
from tools.config_loader import CONFIG
from pathlib import Path
from tools.preprocess_point_cloud import map_angle_to_pixel
from plyfile import PlyData, PlyElement

def image_preprocess(image, img_size):
    """
    Preprocess the image for spherical projection.
    Resizes the image to img_size and converts it to RGB format.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure shape (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # grayscale to RGB

    if image.shape[0] == 3:  # (H, W, 3) -> (3, H, W) 
        image = np.transpose(image, (1, 2, 0))

    # Resize to (271, 720) if needed (zenith: 0–135 at 0.5° → 271 rows)
    if image.shape[0] != img_size[0] or image.shape[1] != img_size[1]:
        image = np.array(Image.fromarray(image).resize((img_size[1], img_size[0]), resample=Image.BILINEAR))

    # Convert to float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)


    print("Image Shape:", image.shape)
    print("Image Dtype:", image.dtype)
    print("Image Max:", image.max())
    return image



def back_project_color_to_ball(df_ball, image, zenith_range= (0, 135), inverse_zenith=False):
    """
    Assigns image color to df_ball using azimuth and zenith angles.
    Assumes image shape (271, 720), angular res = 0.5°, azimuth 0–360, zenith 0–135.
    """
    img_size = int((zenith_range[1] - zenith_range[0]) // 0.5 + 1), 720  # (271, 720) for zenith range (0, 135)
    image = image_preprocess(image, img_size)

    # Compute row, col indices
    zenith_deg_shifted = df_ball['zenith_deg'] - df_ball['zenith_deg'].min()
    row = (zenith_deg_shifted / 0.5).astype(int)
    col = (df_ball['azimuth_deg'] / 0.5).astype(int)
    # Adjust for inverse zenith
    if inverse_zenith:
        row = img_size[0] - 1 - row

    # Get RGB values from image (normalize if uint8)
    rgb = image[row, col] / 255.0 if image.dtype == np.uint8 else image[row, col]

    # Assign to DataFrame
    df_ball = df_ball.copy()
    df_ball['r'] = rgb[:, 0]
    df_ball['g'] = rgb[:, 1]
    df_ball['b'] = rgb[:, 2]

    return df_ball

def get_a_colorized_ball_from_img(image_path: Path, 
                                  key_str: str,
                                  zenith_range: tuple = (0, 135), 
                                  save_dir: Path = None,
                                  visualize: bool = False):
    """
    Get a colorized ball from an image using spherical projection.
    Args:
        image_path (Path): Path to the image.
        key_str (str): Key string to identify the image.
        zenith_range (tuple): Zenith range for the ball.
        save_dir (Path): Directory to save the output point cloud.
    """

    rgb_img = np.array(Image.open(image_path))  # shape (H, W, 3)
    print("RGB Image Shape:", rgb_img.shape)
    print("RGB Image Dtype:", rgb_img.dtype)
    print("RGB Image Max:", rgb_img.max())


    df_colored = back_project_color_to_ball(df_ball, rgb_img, 
                                            zenith_range=zenith_range,
                                            inverse_zenith=False)

    df_cube = create_colored_cube_points(center=[0, 0, 0], size=0.5, samples_per_face=50)
    df_combined = pd.concat([df_colored, df_cube], ignore_index=True)
    visualize_and_save_point_cloud(df_combined, 
                                zenith_range=zenith_range, 
                                visualize=visualize, 
                                save_dir=save_dir,
                                output_prefix=key_str)


def attach_image_colors_to_pcd(rgb_image, channel_names=['r', 'g', 'b']):
    """
    Attach RGB or arbitrary 3-channel color values from a 2D image to a point cloud based on pixel mapping.

    Args:
        rgb_image (np.ndarray): RGB or 3-channel image (H, W, 3).
        channel_names (list): List of 3 strings for output column names. Default: ['r', 'g', 'b'].

    Returns:
        pd.DataFrame: Point cloud DataFrame with additional color columns.
    """

    root_dir = CONFIG["global"]["output_dir"] / 'pcd'
    point_cloud_file = next(root_dir.glob(f"*_color*"), None)
    if point_cloud_file is None:
        point_cloud_file = next(root_dir.glob(f"*_ncr_*"), None)
    

    pc_df = pd.read_csv(point_cloud_file, sep=',')

    if "_color" in point_cloud_file.stem:
        output_file = root_dir / (point_cloud_file.stem + '.csv')
    else:
        file_str = str(point_cloud_file.stem).split("_ncr_")[0]
        output_file = root_dir / (f"{file_str}_color" + '.csv')
        point_cloud_file.unlink()
        print(f"Deleted intermediate file: {point_cloud_file}")
        
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, "Image must have 3 channels"
    assert len(channel_names) == 3, "Exactly 3 channel names must be provided"

    # Map angles to image pixel coordinates
    x_pix, y_pix = map_angle_to_pixel(pc_df['azimuth'], pc_df['elevation'])
    pc_df['x_pix'] = x_pix.astype(int)
    pc_df['y_pix'] = y_pix.astype(int)

    height, width, _ = rgb_image.shape
    colors = []

    for x, y in zip(pc_df['x_pix'], pc_df['y_pix']):
        if 0 <= x < width and 0 <= y < height:
            color = rgb_image[y, x]
        else:
            color = [0, 0, 0]  # Default to black if out of bounds
        colors.append(color)

    color_df = pd.DataFrame(colors, columns=channel_names, index=pc_df.index)
    pc_df = pd.concat([pc_df, color_df], axis=1)

    

    # Drop extra columns used for processing
    pc_df.drop(columns=['x_pix', 'y_pix'], inplace=True)
    pc_df.to_csv(output_file, index=False)
    print(f"✅ Colors {channel_names} attached to point cloud successfully. \nOutput file saved to: {output_file}")


if __name__ == "__main__":
    # ZENITH_RANGE = (75, 105)  # Zenith range for VLP64

    ZENITH_RANGE = (0, 135)  # Zenith range for the CBL V2.0

    df_ball = generate_lidar_ball(radius=10.0, zenith_range=ZENITH_RANGE)

    image_dir1 = Path(CONFIG['global']['output_dir']) / 'pca'
    image_dir2 = Path(CONFIG['global']['output_dir']) / 'img'
    pcd_out_dir = Path(CONFIG['global']['output_dir']) / 'pcd'

    key_str_ls1 = ['PCA_rgb_0_1_2', 'MNF_rgb_0_1_2', 'ICA_rgb_0_1_2']
    key_str_ls2 = ['Roughness-Intensity-Range', 'Roughness-Intensity-Z']
    img_paths = [next(image_dir1.glob(f"*{key_str}*"), None) for key_str in key_str_ls1] + \
                [next(image_dir2.glob(f"*{key_str}*"), None) for key_str in key_str_ls2]


    # Create colorful balls from images
    for image_path, key_str in zip(img_paths, key_str_ls1 + key_str_ls2):
        if image_path is None:
            raise FileNotFoundError(f"No file containing '{key_str}' found in output_dir.")

        get_a_colorized_ball_from_img(image_path, 
                                      key_str,
                                      zenith_range=ZENITH_RANGE, 
                                      save_dir=pcd_out_dir)

    # Attach colors to original point cloud
    rgb_channel_names = [['pca1', 'pca2', 'pca3'],
                         ['mnf1', 'mnf2', 'mnf3'],
                         ['ica1', 'ica2', 'ica3']]
    for image_path, channel_names in zip(img_paths[:3], rgb_channel_names):
        rgb_img = np.array(Image.open(image_path))

        attach_image_colors_to_pcd(rgb_img, channel_names=channel_names)