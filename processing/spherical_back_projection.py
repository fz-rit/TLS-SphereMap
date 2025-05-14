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
from processing.spherical_projection import load_image_cube_and_meta
from numpy.typing import NDArray

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


    # Resize to (271, 720) if needed (zenith: 0–135 at 0.5° → 271 rows)
    if image.dtype == np.float64:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        img_resized = Image.fromarray(image_uint8).resize((img_size[1], img_size[0]))
    else:
        img_resized = Image.fromarray(image).resize((img_size[1], img_size[0]), resample=Image.BILINEAR)

    image = np.array(img_resized)

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
    print("Image Size:", img_size)
    print("Image Shape:", image.shape)
    print("Image Dtype:", image.dtype)
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

def get_a_colorized_ball_from_img(rgb_img: NDArray, 
                                  key_str: str,
                                  zenith_range: tuple = (0, 135), 
                                  save_dir: Path = None,
                                  visualize: bool = False):
    """
    Get a colorized ball from an image using spherical projection.
    Args:
        rgb_img (NDArray): RGB image to be projected, shape (H, W, 3).
        key_str (str): Key string to identify the image.
        zenith_range (tuple): Zenith range for the ball.
        save_dir (Path): Directory to save the output point cloud.
    """

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


def attach_image_colors_to_pcd(rgb_image, pcd_out_dir, 
                               canvas_size: tuple[int, int],
                                angular_res: tuple[int, int],
                                channel_names=['r', 'g', 'b']):
    """
    Attach RGB or arbitrary 3-channel color values from a 2D image to a point cloud based on pixel mapping.

    Args:
        rgb_image (np.ndarray): RGB or 3-channel image (H, W, 3).
        channel_names (list): List of 3 strings for output column names. Default: ['r', 'g', 'b'].

    Returns:
        pd.DataFrame: Point cloud DataFrame with additional color columns.
    """

    point_cloud_file = next(pcd_out_dir.glob(f"*_color*"), None)
    if point_cloud_file is None:
        point_cloud_file = next(pcd_out_dir.glob(f"*_filtered_normaled*"), None)
    

    pc_df = pd.read_csv(point_cloud_file, sep=',')

    if "_color" in point_cloud_file.stem:
        output_file = pcd_out_dir / (point_cloud_file.stem + '.csv')
    else:
        file_str = str(point_cloud_file.stem).split("_filtered_normaled")[0]
        output_file = pcd_out_dir / (f"{file_str}_color" + '.csv')
        point_cloud_file.unlink()
        print(f"Deleted intermediate file: {point_cloud_file}")
        
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, "Image must have 3 channels"
    assert len(channel_names) == 3, "Exactly 3 channel names must be provided"

    # Map angles to image pixel coordinates
    x_pix, y_pix = map_angle_to_pixel(pc_df['azimuth'], 
                                      pc_df['elevation'], 
                                      canvas_size=canvas_size,
                                      angular_res=angular_res)
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
    v_fov = CONFIG['global']['v_fov']
    h_fov = CONFIG['global']['h_fov']
    canvas_width = int(h_fov[1] / CONFIG['global']['h_ang_res_deg'])
    canvas_height = int(v_fov[1] / CONFIG['global']['v_ang_res_deg'])
    canvas_size = (canvas_height, canvas_width)
    angular_res = (CONFIG['global']['v_ang_res_deg'], CONFIG['global']['h_ang_res_deg'])
    zenith_range = CONFIG['global']['v_fov']  # Zenith range for the CBL V2.0
    df_ball = generate_lidar_ball(radius=10.0, zenith_range=zenith_range)
    
    out_dir_ls = CONFIG['global']['output_dir_ls']
    for output_dir in out_dir_ls:
        input_file_stem = output_dir.parent.name
        print(f'#######Processing {input_file_stem}...########')

        image_dir = output_dir / 'img'
        pcd_out_dir = output_dir / 'pcd'

        
        key_str = input_file_stem.split('_')[0] + '_' + input_file_stem.split('_')[-1]
        image_cube_path = image_dir / f'{key_str}_image_cube.npy'
        image_cube, metadata = load_image_cube_and_meta(image_cube_path)

        channel_names = metadata['channel_names']
        
        if channel_names[0]== "True-R":
            group_indices = [
                            [6, 7, 8], 
                            [9, 10, 11], 
                            [12, 13, 14], 
                            [15, 16, 17],
                            [18, 19, 20],]
        elif channel_names[0]== "Intensity Map (raw)":
            group_indices = [[3, 4, 5], 
                            [6, 7, 8], 
                            [9, 10, 11], 
                            [12, 13, 14], 
                            [15, 16, 17]]
        else:
            raise ValueError(f"Weird channel name: {channel_names[0]}, checkout previous script.")
        channel_name_groups = [[channel_names[i] for i in group] for group in group_indices]
        rgb_groups = [image_cube[:,:, group_index] for group_index in group_indices]
        get_a_colorized_ball_from_img(rgb_groups[0], 
                                    key_str,
                                    zenith_range=zenith_range, 
                                    save_dir=pcd_out_dir)
        for channel_name_group, rgb_image in zip(channel_name_groups, rgb_groups):
            attach_image_colors_to_pcd(rgb_image, pcd_out_dir, 
                                       channel_names=channel_name_group, 
                                       canvas_size=canvas_size,
                                       angular_res=angular_res)