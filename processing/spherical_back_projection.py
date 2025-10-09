"""
Spherical back projection: attach 2D image colors to 3D point clouds.
Run with: python run_3d_to_2d_pipeline.py --config <config_file>
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from math import ceil, floor

from tools.illustrate_spherical_projection import (generate_geometric_lidar_ball, 
                                                   visualize_and_save_point_cloud, 
                                                   create_colored_cube_points)
from tools.config_loader import get_config
from tools.preprocess_point_cloud import map_angle_to_pixel
from tools.spherical_projection_helper import load_image_cube_and_meta
from numpy.typing import NDArray

def image_preprocess(image, img_size):
    """Preprocess image for spherical projection."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    if image.dtype == np.float64:
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())
        image_uint8 = (image * 255).astype(np.uint8)
        img_resized = Image.fromarray(image_uint8).resize((img_size[1], img_size[0]))
    elif image.dtype == np.uint8:
        img_resized = Image.fromarray(image).resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")
    
    image = np.array(img_resized)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)
    
    return image



def back_project_color_to_ball(df_ball, image, zenith_range=(0, 135), ang_res=0.5, inverse_zenith=False):
    """Assign image colors to ball using azimuth and zenith angles."""
    img_size = (ceil((zenith_range[1] - zenith_range[0]) / ang_res), ceil(360/ang_res))
    image = image_preprocess(image, img_size)
    
    zenith_deg_shifted = np.array(df_ball['zenith_deg'] - df_ball['zenith_deg'].min())
    row = (zenith_deg_shifted / ang_res).astype(int)
    col = (df_ball['azimuth_deg'] / ang_res).astype(int)
    
    row = np.clip(row, 0, img_size[0] - 1)
    col = np.clip(col, 0, img_size[1] - 1)
    
    if inverse_zenith:
        row = img_size[0] - 1 - row
    
    rgb = image[row, col] / 255.0 if image.dtype == np.uint8 else image[row, col]
    
    df_ball = df_ball.copy()
    df_ball['r'] = rgb[:, 0]
    df_ball['g'] = rgb[:, 1]
    df_ball['b'] = rgb[:, 2]
    
    return df_ball

def get_a_colorized_ball_from_img(rgb_img: NDArray, key_str: str, zenith_range: tuple = (0, 135), 
                                  res_deg: float = 0.5, save_dir: Path = None, 
                                  visualize: bool = False, inverse_zenith: bool = False):
    """Generate colorized ball from image using spherical projection."""
    df_ball = generate_geometric_lidar_ball(radius=10.0, zenith_range=zenith_range, res_deg=res_deg)
    df_colored = back_project_color_to_ball(df_ball, rgb_img, zenith_range=zenith_range,
                                            ang_res=res_deg, inverse_zenith=inverse_zenith)
    df_cube = create_colored_cube_points(center=[0, 0, 0], size=0.3, samples_per_face=50)
    df_combined = pd.concat([df_colored, df_cube], ignore_index=True)
    visualize_and_save_point_cloud(df_combined, zenith_range=zenith_range, 
                                   visualize=visualize, save_dir=save_dir, output_prefix=key_str)


def attach_all_color_groups_to_pcd(image_cube, channel_names, paint_pcd_color_groups, 
                                   pcd_out_dir, input_pcd_key: str, canvas_size: tuple[int, int],
                                   angular_res: tuple[int, int], delete_intermediate: bool = False):
    """Attach all color groups to point cloud at once."""
    
    # Find base point cloud file
    pcd_files = list(pcd_out_dir.glob(f"*{input_pcd_key}*.txt"))
    base_pcd_file = next((f for f in pcd_files if "_color" not in f.stem), None)
    
    if base_pcd_file is None:
        raise FileNotFoundError(f"No base point cloud file found with key '{input_pcd_key}' in {pcd_out_dir}")
    
    print(f"📁 Using base file: {base_pcd_file.name}")
    pc_df = pd.read_csv(base_pcd_file, sep=',')
    
    # Map angles to pixels once
    x_pix, y_pix = map_angle_to_pixel(pc_df['azimuth'], pc_df['elevation'], 
                                      canvas_size=canvas_size, angular_res=angular_res)
    x_pix, y_pix = x_pix.astype(int), y_pix.astype(int)
    
    height, width, _ = image_cube.shape
    all_color_columns = {}
    
    # Process all color groups
    for i, color_group in enumerate(paint_pcd_color_groups, 1):
        print(f"🖌️ [{i}/{len(paint_pcd_color_groups)}] Processing: {color_group}")
        
        rgb_image = image_cube[:, :, [channel_names.index(name) for name in color_group]]
        colors = []
        
        for x, y in zip(x_pix, y_pix):
            if 0 <= x < width and 0 <= y < height:
                colors.append(rgb_image[y, x])
            else:
                colors.append([0, 0, 0])
        
        color_array = np.array(colors)
        for j, channel_name in enumerate(color_group):
            all_color_columns[channel_name] = color_array[:, j]
    
    # Add all colors at once
    color_df = pd.DataFrame(all_color_columns, index=pc_df.index)
    pc_df = pd.concat([pc_df, color_df], axis=1)
    
    # Save output
    file_prefix = base_pcd_file.stem.split(input_pcd_key)[0] if input_pcd_key in base_pcd_file.stem else base_pcd_file.stem
    output_file = pcd_out_dir / f"{file_prefix}color.csv"
    
    if delete_intermediate:
        base_pcd_file.unlink()
        print(f"🗑️ Deleted: {base_pcd_file.name}")
    
    pc_df.to_csv(output_file, index=False)
    print(f"✅ Added {len(all_color_columns)} color columns: {list(all_color_columns.keys())}")
    print(f"📄 Output: {output_file}")
    
    return str(output_file)


def prepare_color_group(channel_names, color_group):
    """Prepare and validate color groups for point cloud painting."""
    paint_pcd_color_group_dict = {
        'izr': ['intensity_adjusted', 'z_adjusted', 'range_adjusted'],
        'normals': ['Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn'],
        'pca': ['PCA1', 'PCA2', 'PCA3'],
        'true_rgb': ['True-R', 'True-G', 'True-B']
    }
    
    for color in color_group:
        if color not in paint_pcd_color_group_dict:
            raise ValueError(f"Color group {color} not recognized. Available: {list(paint_pcd_color_group_dict.keys())}")
    
    paint_pcd_color_groups = [paint_pcd_color_group_dict[group] for group in color_group]
    
    for group in paint_pcd_color_groups:
        for name in group:
            if name not in channel_names:
                raise ValueError(f"Channel {name} not found. Available: {channel_names}")
    
    print(f"Using color groups: {paint_pcd_color_groups}")
    return paint_pcd_color_groups

if __name__ == "__main__":
    current_config = get_config()
    if current_config is None:
        print("❌ Error: Configuration not loaded. Please run through run_3d_to_2d_pipeline.py")
        exit(1)

    # Load configuration
    global_params = current_config['global']
    v_fov, h_fov = global_params['v_fov'], global_params['h_fov']
    angular_res = (global_params['v_ang_res_deg'], global_params['h_ang_res_deg'])
    canvas_size = global_params['canvas_size']
    delete_intermediate_file = global_params['delete_intermediate_file']
    out_dir_ls = global_params['output_dir_ls']
    input_path_ls = global_params['input_path_ls']
    out_signature_str = current_config['calc_geom_feature']['out_signature_str']
    inverse_zenith = current_config['calc_geom_feature']['flip_mangrove']
    generate_virtual_ball = current_config['back_projection']['generate_virtual_ball']
    paint_color_groups = current_config['back_projection']['paint_pcd_color_groups']

    for input_file, output_dir in zip(input_path_ls, out_dir_ls):
        print(f'### Processing {input_file.stem} ###')

        image_dir = output_dir / 'img'
        pcd_out_dir = output_dir / 'pcd'
        
        image_cube_path = image_dir / f'{input_file.stem}_image_cube.npy'
        image_cube, metadata = load_image_cube_and_meta(image_cube_path)
        channel_names = metadata['channel_names']
        paint_pcd_color_groups = prepare_color_group(channel_names, paint_color_groups)

        # Attach all color groups at once
        print(f"🎨 Processing {len(paint_pcd_color_groups)} color groups")
        attach_all_color_groups_to_pcd(
            image_cube=image_cube, channel_names=channel_names,
            paint_pcd_color_groups=paint_pcd_color_groups, pcd_out_dir=pcd_out_dir,
            input_pcd_key=out_signature_str, canvas_size=canvas_size,
            angular_res=angular_res, delete_intermediate=delete_intermediate_file
        )

        # Generate virtual balls if enabled
        if generate_virtual_ball:
            print(f"🌐 Generating {len(paint_pcd_color_groups)} virtual balls...")
            for color_group in paint_pcd_color_groups:
                rgb_image = image_cube[:, :, [channel_names.index(name) for name in color_group]]
                ball_key_str = f"colorball_{'_'.join(color_group)}"
                get_a_colorized_ball_from_img(rgb_image, ball_key_str, res_deg=angular_res[0],
                                            zenith_range=v_fov, save_dir=pcd_out_dir, 
                                            inverse_zenith=inverse_zenith)
        