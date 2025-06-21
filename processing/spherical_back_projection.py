"""
Run this file with `python run_3d_to_2d_pipeline.py` or `python -m processing.spherical_back_projection`

"""

import numpy as np
from tools.illustrate_spherical_projection import (generate_geometric_lidar_ball, 
                                             visualize_and_save_point_cloud, 
                                             create_colored_cube_points)
from PIL import Image
import pandas as pd
from tools.config_loader import CONFIG
from pathlib import Path
from tools.preprocess_point_cloud import map_angle_to_pixel
from plyfile import PlyData, PlyElement
from tools.spherical_projection_helper import load_image_cube_and_meta
from numpy.typing import NDArray
from math import ceil

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
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())
        image_uint8 = (image * 255).astype(np.uint8)
        img_resized = Image.fromarray(image_uint8).resize((img_size[1], img_size[0]))
    elif image.dtype == np.uint8:
        img_resized = Image.fromarray(image).resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}. Expected uint8 or float64.")

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
                                  visualize: bool = False,
                                  inverse_zenith: bool = False):
    """
    Get a colorized ball from an image using spherical projection.
    Args:
        rgb_img (NDArray): RGB image to be projected, shape (H, W, 3).
        key_str (str): Key string to identify the image.
        zenith_range (tuple): Zenith range for the ball.
        save_dir (Path): Directory to save the output point cloud.
    """
    df_ball = generate_geometric_lidar_ball(radius=10.0, zenith_range=zenith_range)
    df_colored = back_project_color_to_ball(df_ball, rgb_img, 
                                            zenith_range=zenith_range,
                                            inverse_zenith=inverse_zenith)

    df_cube = create_colored_cube_points(center=[0, 0, 0], size=0.5, samples_per_face=50)
    df_combined = pd.concat([df_colored, df_cube], ignore_index=True)
    visualize_and_save_point_cloud(df_combined, 
                                zenith_range=zenith_range, 
                                visualize=visualize, 
                                save_dir=save_dir,
                                output_prefix=key_str)


def attach_image_colors_to_pcd(rgb_image, pcd_out_dir, input_pcd_key: str,
                               canvas_size: tuple[int, int],
                                angular_res: tuple[int, int],
                                channel_names=['r', 'g', 'b'],
                                delete_intermediate: bool = False):
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
        point_cloud_file = next(pcd_out_dir.glob(f"*{input_pcd_key}*"), None)

    pc_df = pd.read_csv(point_cloud_file, sep=',')

    if "_color" in point_cloud_file.stem:
        output_file = pcd_out_dir / (point_cloud_file.stem + '.csv')
    else:
        file_str = point_cloud_file.stem.split(input_pcd_key)[0]
        output_file = pcd_out_dir / (f"{file_str}color" + '.csv')
        if delete_intermediate:
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


def prepare_color_group(channel_names, color_group):
    """
    Prepare color groups for painting the point cloud.

    Args:
        channel_names (list): List of channel names in the image cube.
        color_group (list): List of color groups to prepare.

    Returns:
        list: List of prepared color groups.
    """

    paint_pcd_color_group_dict = {'izr': ['intensity_adjusted', 'z_adjusted', 'range_adjusted'],
                                  'normals': ['Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn'],
                                  'pca': ['PCA1', 'PCA2', 'PCA3'],
                                  'true_rgb': ['True-R', 'True-G', 'True-B']}

    c_keys = list(paint_pcd_color_group_dict.keys())
    for color in color_group:
        if color not in c_keys:
            raise ValueError(f"Color group {color} not recognized. Available groups: {c_keys}")
    paint_pcd_color_groups = [paint_pcd_color_group_dict[group] for group in color_group]

    for group in paint_pcd_color_groups:
        for name in group:
            if name not in channel_names:
                raise ValueError(f"Channel name {name} not found in image cube metadata. Available channels: {channel_names}")

    print(f"Using color groups: {paint_pcd_color_groups}")
    return paint_pcd_color_groups

if __name__ == "__main__":

    # Load configuration
    global_params = CONFIG['global']
    v_fov = global_params['v_fov']
    h_fov = global_params['h_fov']
    angular_res = (global_params['v_ang_res_deg'], global_params['h_ang_res_deg'])
    canvas_size = global_params['canvas_size']
    delete_intermediate_file = global_params['delete_intermediate_file']
    out_dir_ls = global_params['output_dir_ls']
    out_signature_str = CONFIG['calc_geom_feature']['out_signature_str']
    inverse_zenith = CONFIG['calc_geom_feature']['flip_mangrove']
    generate_virtual_ball = CONFIG['back_projection']['generate_virtual_ball']
    paint_color_groups = CONFIG['back_projection']['paint_pcd_color_groups']

    for output_dir in out_dir_ls:
        input_file_stem = output_dir.parent.name
        print(f'#######Processing {input_file_stem}...########')

        image_dir = output_dir / 'img'
        pcd_out_dir = output_dir / 'pcd'
        
        key_str = input_file_stem.split('_')[0] + '_' + input_file_stem.split('_')[-1]
        image_cube_path = image_dir / f'{key_str}_image_cube.npy'
        image_cube, metadata = load_image_cube_and_meta(image_cube_path)
        channel_names = metadata['channel_names']
        paint_pcd_color_groups = prepare_color_group(channel_names, paint_color_groups)

        # Paint the point cloud with different color groups
        for color_group in paint_pcd_color_groups:
            rgb_image = image_cube[:, :, [channel_names.index(name) for name in color_group]]
            attach_image_colors_to_pcd(rgb_image, pcd_out_dir, out_signature_str,
                                       channel_names=color_group, 
                                       canvas_size=canvas_size,
                                       angular_res=angular_res,
                                       delete_intermediate=delete_intermediate_file)

            if generate_virtual_ball:
                str_ls = input_file_stem.split('_')
                pcd_signature_str = f"pcd_{str_ls[2]}_{str_ls[-1][-4:]}"
                ball_key_str = f"{pcd_signature_str}_{'_'.join(color_group)}"
                get_a_colorized_ball_from_img(rgb_image,
                                        ball_key_str,
                                        zenith_range=v_fov, 
                                        save_dir=pcd_out_dir,
                                        inverse_zenith=inverse_zenith)
        