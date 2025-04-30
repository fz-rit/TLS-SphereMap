# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from tools.hdr_adjustments import contrast_enhancement
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.colors import ListedColormap, BoundaryNorm
from tools.preprocess_point_cloud import map_angle_to_pixel
from typing import Union, List, Dict, Any
from pathlib import Path
from tools.plot_tools import display_unwrapped_single_band_images, display_unwrapped_rgb_image, display_single_band_img_wt_discrete_values
from tools.norm_to_hsv import attach_normal_color_to_df
from tools.config_loader import CONFIG
import json
from tools.pcd_utils import create_dir_if_not_exists

# Ignore warnings
pd.options.mode.chained_assignment = None


CANVAS_WIDTH = 1440 
CANVAS_HEIGHT = 540

def load_and_preprocess_point_cloud(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load the point cloud data and preprocess it by:
    1. Attach the pixel values for azimuth and elevation
    2. Attach the HSV color values for the normal
    3. Reverse the range values and z values

    Args:
        filename: str or Path: The path to the point cloud data

    Returns:
        pd.DataFrame: The preprocessed point cloud data
    """

    df_filtered = pd.read_csv(filename, sep=',')
    azimuth, elevation = df_filtered['azimuth'], df_filtered['elevation']
    x_pix, y_pix = map_angle_to_pixel(azimuth, elevation)
    df_filtered['x_pix'] = x_pix
    df_filtered['y_pix'] = y_pix

    # Grab HSV color from the normal and attach the color to the dataframe
    df_filtered_ncolored = attach_normal_color_to_df(df_filtered)
    
    # shift the z values to start from 0
    z_min = df_filtered_ncolored['Z'].min()
    df_filtered_ncolored['Z'] = - (df_filtered_ncolored['Z'] - z_min)
    print("Z shifted to start from 0.")

    # # Normalize columns to (0.01, 1.0): Intensity, Z, curvature, roughness
    for col_name in ['Intensity', 'Z', 'curvature', 'roughness']:
        col_min = df_filtered_ncolored[col_name].min()
        col_max = df_filtered_ncolored[col_name].max()
        df_filtered_ncolored[col_name] = 0.01 + 0.99 * (df_filtered_ncolored[col_name] - col_min) / (col_max - col_min)
        print(f"{col_name} normalized to (0.01, 1.0).")

    
    # Check if there are NaN values or negative values in the columns of interest
    for col_name in ['Intensity', 'azimuth', 'zenith', 'range1metres']:
            assert df_filtered_ncolored[col_name].isnull().sum() == 0, f"Column {col_name} has NaN values"
            assert (df_filtered_ncolored[col_name] < 0).sum() == 0, f"Column {col_name} has negative values"

    return df_filtered_ncolored


def unwrap_point_cloud_to_2d_images(filename: str) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Unwrap the point cloud to 2D images with x being azimuth angle, y being zenith angle, and pixel value with different kinds of scalar fields.
    Including density, intensity, range, and range-xy images.

    Parameters:
    filename (str): The path to the point cloud data file in CSV format.
    saveflag (bool): If True, save the images to the output_dir directory. Default is True.

    Returns:
    df_filtered (pd.DataFrame): A DataFrame containing the filtered and processed point cloud data with additional columns for pixel coordinates.
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the density image, adjusted intensity image, adjusted range image, and adjusted range-xy image.
    """
    ## Load and Preprocess the point cloud data
    df_filtered_ncolored = load_and_preprocess_point_cloud(filename)
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False)

    # Compute intensity and range per pixel. 
    # [Since less than 0.1% of the pixels contain more than one point, 
    # it may or may not matter if we use mean or max.]

    # Use mean to avoid too much noise

    group_intensity = grouped['Intensity'].mean()
    group_z = grouped['Z'].min()
    group_range = grouped['range1metres'].mean()
    pts_per_pixel = grouped.size()
    curvature_per_pixel = grouped['curvature'].mean()
    roughness_per_pixel = grouped['roughness'].mean()
    

    ## Map the computed values to the image arrays
    ## Create empty image canvases with proper resolution
    intensity_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    range_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    z_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    density_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    curvature_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    roughness_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    
    pxpy_indices = np.array(group_intensity.index.tolist())
    intensity_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = group_intensity.values
    z_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = group_z.values
    range_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = group_range.values
    density_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = pts_per_pixel.values
    curvature_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = curvature_per_pixel.values
    roughness_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = roughness_per_pixel.values


    # Apply HDR adjustment to intensity and range images
    intensity_image_adjusted = contrast_enhancement(intensity_image, stretch_percentile=0.1)
    z_image_adjusted = contrast_enhancement(z_image, stretch_percentile=0.1)
    range_image_adjusted = contrast_enhancement(range_image, stretch_percentile=0)
    curvature_image_adjusted = contrast_enhancement(curvature_image)
    roughness_image_adjusted = contrast_enhancement(roughness_image)

    image_names = ['Density Map', 
                    'Intensity Map (adjusted)', 
                    'Z Map Inverse (adjusted)',
                    'Range Map (adjusted)', 
                    'Curvature Map (adjusted)',
                    'Roughness Map (adjusted)',
                    'Intensity Map (raw)', 
                    'Z Map Inverse (raw)',
                    'Range Map (raw)', 
                    'Curvature Map (raw)',
                    'Roughness Map (raw)',
                    ]
    
    

    output_images = (density_image, 
                     intensity_image_adjusted, 
                     z_image_adjusted,
                     range_image_adjusted, 
                     curvature_image_adjusted,
                     roughness_image_adjusted,
                     intensity_image, 
                     z_image,
                     range_image, 
                     curvature_image,
                     roughness_image,
                     )
    
    output_images_dict = {image_name: image for image_name, image in zip(image_names, output_images)}
   
    return df_filtered_ncolored, output_images_dict


def unwrap_pc_normals_to_rgb_image(df_filtered_ncolored: pd.DataFrame) -> np.ndarray:
    """
    Unwrap the point cloud to 2D images with x being azimuth angle, y being zenith angle, and pixel values
    colorized by the normal vectors.

    Parameters:
    
    """
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False)

    r_per_pixel = grouped['n_r'].mean()
    g_per_pixel = grouped['n_g'].mean()
    b_per_pixel = grouped['n_b'].mean()


    ## Map the computed values to the image arrays
    ## Create empty image canvases with proper resolution
    r_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    g_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    b_image = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    
    pxpy_indices = np.array(r_per_pixel.index.tolist())
    r_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = r_per_pixel.values
    g_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = g_per_pixel.values
    b_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = b_per_pixel.values

    # Stack the single channel images to create an RGB image
    rgb_image = np.stack((r_image, g_image, b_image), axis=-1) # Shape: (540, 1440, 3)
   
    return rgb_image



def save_image_cube_and_metadata(
    output_images_dict: Dict[str, np.ndarray],
    normals_rgb_image: np.ndarray,
    output_dir: Path,
    key_str: str,
    input_folder: str
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Saves an image cube and its metadata.

    This function constructs an image cube by stacking different adjusted maps
    (Intensity, Z, Range, Curvature, Roughness) along with the normal vector RGB
    image and then saves it as a `.npy` file. Additionally, metadata related to
    the image processing steps and other relevant information is saved in a separate
    `.npy` file.

    Parameters:
    - output_images_dict: A dictionary mapping titles of processed maps to numpy arrays.
    - normals_rgb_image: A numpy array representing the RGB image of normal vectors.
    - output_dir: The directory where the output files will be saved.
    - key_str: A string to be included in the filenames for the saved files.
    - input_folder: Another string that will be included in the metadata.

    Returns:
    - None
    """

    # Titles for the image channels
    save_titles = [
        'Intensity Map (adjusted)',
        'Z Map Inverse (adjusted)',
        'Range Map (adjusted)', 
        'Curvature Map (adjusted)',
        'Roughness Map (adjusted)'
    ]

    # Dynamically fetch the adjusted maps
    adjusted_maps = []
    for title in save_titles:
        try:
            adjusted_map = output_images_dict[title]
            # Examine the shape of the adjusted map
            if adjusted_map.shape != normals_rgb_image.shape[:2]:
                raise ValueError(f"Shape mismatch: {title} shape: {adjusted_map.shape}, normals_rgb_image shape: {normals_rgb_image.shape[:2]}")
            adjusted_maps.append(output_images_dict[title])
        except KeyError as e:
            print(f"Missing key: {e}")
            raise KeyError(f"Key {e} not found in the output_images_dict.")

    image_cube = np.stack(adjusted_maps, axis=-1)
    image_cube = np.concatenate([image_cube, normals_rgb_image], axis=-1)

    # Save the image cube
    image_cube_path = output_dir / f'{key_str}_image_cube.npy'
    np.save(image_cube_path, image_cube)
    print(f"Image cube saved to {image_cube_path}, shape: {image_cube.shape}")

    # Prepare metadata dynamically
    preprocess_metadata = {}
    for title in save_titles:
        preprocess_metadata[title] = f"Contrast enhanced {title.lower()}; first normalized all the values to (0.1, 1.0), then did histogram equalization on the valid pixels."

    # Include the metadata for the normals RGB map
    preprocess_metadata['Normals RGB - Cnx, Cny, Cnz'] = 'RGB image colorized by the normal vectors.'

    # Metadata for the image cube
    metadata = {
        'titles': save_titles + ['Normals RGB - Cnx, Cny, Cnz'],
        'shape': image_cube.shape,
        'dtype': image_cube.dtype,
        'key_str': key_str,
        'input_folder': input_folder,
        'output_dir': output_dir,
        'preprocess_metadata': preprocess_metadata
    }

    # Save the metadata
    metadata_path = output_dir / f'{key_str}_image_cube_metadata.json'
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4, default=str)
    print(f"Metadata saved to {metadata_path}")
    print(f"Metadata titles:\n{metadata['titles']}")

    return image_cube, metadata



def load_image_cube_and_metadata(image_cube_path: Path, metadata_path: Path) -> Dict[str, Any]:
    """
    Loads an image cube and its metadata from saved .npy files.

    Parameters:
    - image_cube_path: The path to the saved image cube file.
    - metadata_path: The path to the saved metadata file.

    Returns:
    - A dictionary containing the image cube and metadata.

    # # Example usage of the load_image_cube_and_metadata function
    # # Define file paths
    # image_cube_path = output_dir / f'{input_file_stem}_image_cube.npy'
    # metadata_path = output_dir / f'{input_file_stem}_image_cube_metadata.npy'

    # # Load the image cube and metadata
    # data = load_image_cube_and_metadata(image_cube_path, metadata_path)

    # # Access the image cube and metadata separately
    # image_cube = data['image_cube']
    # metadata = data['metadata']

    # # Example: print out the shape of the image cube and titles from the metadata
    # print(f"Image Cube Shape: {image_cube.shape}")
    # print("Metadata Titles: ", metadata['titles'])
    """
    
    # Load the image cube (8-channel data)
    image_cube = np.load(image_cube_path)
    print(f"Image cube loaded from {image_cube_path}, shape: {image_cube.shape}")
    
    # Load the metadata
    metadata = np.load(metadata_path, allow_pickle=True).item()
    print(f"Metadata loaded from {metadata_path}")

    return {
        'image_cube': image_cube,
        'metadata': metadata
    }



def normalize_and_stack_images(image_list: List[np.array], method="global"):
    """
    Normalize a list of image channels either globally after stacking or per-channel before stacking.

    Parameters:
        image_list (list of np.ndarray): List of 2D arrays representing image channels (e.g., intensity, range).
        method (str): Normalization method, either "global" for global normalization after stacking
                      or "per_channel" for per-channel normalization before stacking.

    Returns:
        np.ndarray: Normalized and stacked 3D array with shape (H, W, C).
    """
    if method not in ["global", "per_channel"]:
        raise ValueError("Invalid method. Choose 'global' or 'per_channel'.")

    if method == "per_channel":
        # Normalize each channel independently
        for i in range(len(image_list)):
            img_mean = image_list[i].mean()
            img_std = image_list[i].std()
            img_norm = (image_list[i] - img_mean) / (img_std + 1e-8)  # Avoid division by zero
            img_min = img_norm.min()
            img_max = img_norm.max()
            image_list[i] = (img_norm - img_min) / (img_max - img_min)

        # Stack the normalized channels
        stacked_image = np.stack(image_list, axis=-1)

    elif method == "global":
        # Normalize each channel first (zero mean, unit variance)
        for i in range(len(image_list)):
            img_mean = image_list[i].mean()
            img_std = image_list[i].std()
            image_list[i] = (image_list[i] - img_mean) / (img_std + 1e-8)

        # Stack the channels
        stacked_image = np.stack(image_list, axis=-1)

        # Perform global rescaling to (0, 1)
        stack_min = stacked_image.min()
        stack_max = stacked_image.max()
        stacked_image = (stacked_image - stack_min) / (stack_max - stack_min + 1e-8)

    return stacked_image


def create_pseudo_rgb_image(intensity_image: np.ndarray, 
                            range_image: np.ndarray, 
                            third_channel_img: np.ndarray, 
                            figure_title: str = '',
                            output_dir:Path=None,
                            saveflag:bool=False,
                            visualize: bool = True) -> np.ndarray:
    """
    Combine intensity, range, and density images into a pseudo-RGB image.

    Parameters:
    -----------
    intensity_image : np.ndarray
        The adjusted intensity image.
    range_image : np.ndarray
        The adjusted range image.
    third_channel_img : np.ndarray
        The third channel image, can be roughness or curvature.

    Returns:
    --------
    np.ndarray
        The pseudo-RGB image.
    """
    # Normalize each channel before stacking them
    image_list = [intensity_image, range_image, third_channel_img]
    pseudo_rgb_image = normalize_and_stack_images(image_list, method="global")
    display_unwrapped_rgb_image(pseudo_rgb_image, 
                            figure_title, 
                            output_dir, 
                            saveflag,
                            visualize
                            )

    return pseudo_rgb_image


def main():
    global_params = CONFIG['global']
    input_file_stem = global_params['input_file_stem']
    input_folder = global_params['input_folder']
    params = CONFIG['spherical_projection']
    saveflag = params['saveflag']
    visualize = params['visualize']
    output_dir = Path(global_params['output_dir'])
    pcd_dir = output_dir / 'pcd'
    img_out_dir = output_dir / 'img'
    create_dir_if_not_exists(img_out_dir)
    filename = next(pcd_dir.glob("*_ncr_*"), None)
    if filename is None:
        raise FileNotFoundError("No file containing '_filtered_' found in output_dir.")


    key_str = input_file_stem.split('_')[0] + '_' + input_file_stem.split('_')[-1]
    df_filtered, output_images_dict = unwrap_point_cloud_to_2d_images(filename)
    density_image = output_images_dict['Density Map']
    display_single_band_img_wt_discrete_values(density_image, 
                                            title='Point Density Map', 
                                            output_dir=img_out_dir, 
                                            saveflag=saveflag,
                                            visualize=visualize)


    # Display and save the adjusted intensity and range images
    titles = ['Intensity Map (adjusted)', 
            'Z Map Inverse (adjusted)',
            'Range Map (adjusted)', 
            'Curvature Map (adjusted)',
            'Roughness Map (adjusted)',
            'Intensity Map (raw)', 
            'Z Map Inverse (raw)',
            'Range Map (raw)', 
            'Curvature Map (raw)',
            'Roughness Map (raw)',
            ]
    display_images = [output_images_dict[title] for title in titles]
    display_unwrapped_single_band_images(display_images, 
                                        titles=titles,
                                        key_str=key_str,
                                        output_dir=img_out_dir,
                                        saveflag=saveflag,
                                        visualize=visualize
                                        )

    # Display the Pseudo-RGB image from normals.
    normals_rgb_image = unwrap_pc_normals_to_rgb_image(df_filtered)
    display_unwrapped_rgb_image(normals_rgb_image, 
                                figure_title=f'HSV_colorized_map_from_normals_{key_str}', 
                                saveflag=saveflag, 
                                output_dir=img_out_dir,
                                visualize=visualize
                                )

    # Save the image cube and metadata
    save_image_cube_and_metadata(
                                output_images_dict, 
                                normals_rgb_image, 
                                img_out_dir, 
                                key_str, 
                                input_folder
                                )

    # Create pseudo-RGB images from various combinations of intensity, range, and roughness, and Z.
    figure_title_1 = f'Pseudo-RGB_Intensity-Z-Roughness-{key_str}'
    figure_title_2 = f'Pseudo-RGB_Intensity-Range-Roughness-{key_str}'
    figure_title_3 = f'Pseudo-RGB_Z-Roughness-Intensity-{key_str}'
    figure_title_4 = f'Pseudo-RGB_Roughness-Intensity-Z-{key_str}'
    figure_title_5 = f'Pseudo-RGB_Roughness-Intensity-Range-{key_str}'
    figure_title_6 = f'Pseudo-RGB_Intensity-Range-Z-{key_str}'

    intensity_image_adjusted = output_images_dict['Intensity Map (adjusted)']
    z_image_adjusted = output_images_dict['Z Map Inverse (adjusted)']
    range_image_adjusted = output_images_dict['Range Map (adjusted)']
    roughness_image_adjusted = output_images_dict['Roughness Map (adjusted)']
    shuffle_images = [intensity_image_adjusted, z_image_adjusted, range_image_adjusted, roughness_image_adjusted]
    shuffle_orders = [[0, 1, 3], [0, 2, 3], [1, 3, 0], [3, 0, 1], [3, 0, 2], [0, 2, 1]]
    figure_titles = [figure_title_1, figure_title_2, figure_title_3, figure_title_4, figure_title_5, figure_title_6]
    for (shuffle_order, figure_title) in zip(shuffle_orders, figure_titles):
        pseudo_rgb_image = create_pseudo_rgb_image(shuffle_images[shuffle_order[0]], 
                                                    shuffle_images[shuffle_order[1]], 
                                                    shuffle_images[shuffle_order[2]], 
                                                    figure_title=figure_title, 
                                                    output_dir=img_out_dir, 
                                                    saveflag=saveflag, 
                                                    visualize=visualize)

if __name__ == "__main__":
    main()