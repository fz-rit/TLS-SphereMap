
import pandas as pd
import numpy as np
from tools.hdr_adjustments import contrast_enhancement
from tools.preprocess_point_cloud import map_angle_to_pixel
from typing import Union, List, Dict, Any
from pathlib import Path
from tools.plot_tools import plot_correlation_matrix, plot_pca_components, plot_rgb_permutations
from tools.pca_helper import compute_band_correlation, compute_pca_components, compute_mnf, compute_ica, z_score_standardize
from tools.plot_tools import display_unwrapped_single_band_images, display_unwrapped_rgb_image, display_single_band_img_wt_discrete_values, histogram_to_ascii
from tools.norm_to_hsv import attach_normal_color_to_df
from tools.config_loader import CONFIG
from tools.pcd_utils import create_dir_if_not_exists
import yaml
from PIL import Image

pd.options.mode.chained_assignment = None

def load_and_preprocess_point_cloud(filename: Union[str, Path], 
                                    canvas_size: tuple[int, int],
                                    angular_res: tuple[int, int]) -> pd.DataFrame:
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
    x_pix, y_pix = map_angle_to_pixel(azimuth, elevation, canvas_size, angular_res)
    df_filtered['x_pix'] = x_pix
    df_filtered['y_pix'] = y_pix

    # Grab HSV color from the normal and attach the color to the dataframe
    df_filtered_ncolored = attach_normal_color_to_df(df_filtered)
    
    # shift the z values to start from 0
    z_min = df_filtered_ncolored['Z'].min()
    df_filtered_ncolored['Z'] = - (df_filtered_ncolored['Z'] - z_min)
    print("Z shifted to start from 0.")

    # # Normalize columns to (0.01, 1.0): Intensity, Z
    for col_name in ['Intensity', 'Z']:
        col_min = df_filtered_ncolored[col_name].min()
        col_max = df_filtered_ncolored[col_name].max()
        df_filtered_ncolored[col_name] = 0.01 + 0.99 * (df_filtered_ncolored[col_name] - col_min) / (col_max - col_min)
        print(f"{col_name} normalized to (0.01, 1.0).")

    
    # Check if there are NaN values or negative values in the columns of interest
    for col_name in ['Intensity', 'azimuth', 'zenith', 'rangemeter']:
            assert df_filtered_ncolored[col_name].isnull().sum() == 0, f"Column {col_name} has NaN values"
            assert (df_filtered_ncolored[col_name] < 0).sum() == 0, f"Column {col_name} has negative values"

    return df_filtered_ncolored

def equirectangular_projection_multi(filename: str,
                                    canvas_size: tuple[int, int],
                                    angular_res: tuple[int, int],
                                    dataset_name: str = 'MANGROVE',
                                    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Unwrap the point cloud to 2D images with x being azimuth angle, y being zenith angle, and pixel value with different kinds of scalar fields.
    Including density, intensity, range, and range-xy images.

    Parameters:
    filename (str): The path to the point cloud data file in CSV format.

    Returns:
    df_filtered (pd.DataFrame): A DataFrame containing the filtered and processed point cloud data with additional columns for pixel coordinates.
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the density image, adjusted intensity image, adjusted range image, and adjusted range-xy image.
    """
    ## Load and Preprocess the point cloud data

    df_filtered_ncolored = load_and_preprocess_point_cloud(filename,
                                                           canvas_size=canvas_size,
                                                           angular_res=angular_res)

    if dataset_name == 'SEMANTIC3D':
        cols_to_convert = ['rangemeter', 'r', 'g', 'b']
        for col in cols_to_convert:
            if col in df_filtered_ncolored.columns:
                df_filtered_ncolored[col] = df_filtered_ncolored[col].astype(float)

        # Group by pixel coordinates and calculate the required values
        grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).agg(
            Intensity=('Intensity', 'mean'),
            Z=('Z', 'min'),
            rangemeter=('rangemeter', 'mean'),
            curvature=('curvature', 'mean'),
            # roughness=('roughness', 'mean'),
            anisotropy=('anisotropy', 'mean'),
            # surface_variation=('surface_variation', 'mean'),
            planarity=('planarity', 'mean'),
            r=('r', 'mean'),  # Add r, g, b to the aggregation
            g=('g', 'mean'),
            b=('b', 'mean'),
            pts_per_pixel=('y_pix', 'size')  # Use any column to count points per pixel
        )
        rgb_image = np.full((*canvas_size, 3), 255, dtype=np.uint8) #create the rgb image here
        rgb_image[y_indices, x_indices, 0] = grouped['r'].values
        rgb_image[y_indices, x_indices, 1] = grouped['g'].values
        rgb_image[y_indices, x_indices, 2] = grouped['b'].values

    elif dataset_name == 'MANGROVE':
        # Group by pixel coordinates and calculate the required values
        grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).agg(
            Intensity=('Intensity', 'mean'),
            Z=('Z', 'min'),
            rangemeter=('rangemeter', 'mean'),
            curvature=('curvature', 'mean'),
            # roughness=('roughness', 'mean'),
            anisotropy=('anisotropy', 'mean'),
            # surface_variation=('surface_variation', 'mean'),
            planarity=('planarity', 'mean'),
            pts_per_pixel=('y_pix', 'size')  # Use any column to count points per pixel
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are 'SEMANTIC3D' and 'MANGROVE'.")
    # Create empty image canvases with proper resolution
    intensity_image = np.zeros(canvas_size, dtype=np.float32)
    range_image = np.zeros(canvas_size, dtype=np.float32)
    z_image = np.zeros(canvas_size, dtype=np.float32)
    density_image = np.zeros(canvas_size, dtype=np.float32)
    curvature_image = np.zeros(canvas_size, dtype=np.float32)
    anisotropy_image = np.zeros(canvas_size, dtype=np.float32)
    planarity_image = np.zeros(canvas_size, dtype=np.float32)
    # Use vectorized assignment to map the computed values to the image arrays
    y_indices = grouped.index.get_level_values(0)
    x_indices = grouped.index.get_level_values(1)

    intensity_image[y_indices, x_indices] = grouped['Intensity'].values
    z_image[y_indices, x_indices] = grouped['Z'].values
    range_image[y_indices, x_indices] = grouped['rangemeter'].values
    density_image[y_indices, x_indices] = grouped['pts_per_pixel'].values
    curvature_image[y_indices, x_indices] = grouped['curvature'].values
    # roughness_image[y_indices, x_indices] = grouped['roughness'].values
    anisotropy_image[y_indices, x_indices] = grouped['anisotropy'].values
    # surface_variation_image[y_indices, x_indices] = grouped['surface_variation'].values
    planarity_image[y_indices, x_indices] = grouped['planarity'].values


    # Apply HDR adjustment to intensity and range images
    intensity_image_adjusted = contrast_enhancement(intensity_image, stretch_percentile=0.1)
    z_image_adjusted = contrast_enhancement(z_image, stretch_percentile=0)
    range_image_adjusted = contrast_enhancement(range_image, stretch_percentile=0)
    # Normalize geometric feature map to [0.0, 1.0]
    curvature_image = (curvature_image - curvature_image.min()) / (curvature_image.max() - curvature_image.min() + 1e-8)
    anisotropy_image = (anisotropy_image - anisotropy_image.min()) / (anisotropy_image.max() - anisotropy_image.min() + 1e-8)
    planarity_image = (planarity_image - planarity_image.min()) / (planarity_image.max() - planarity_image.min() + 1e-8)

    image_names = ['Density',
                    'Intensity (adjusted)',
                    'Z-Inv (adjusted)',
                    'Range (adjusted)',
                    'Intensity (raw)',
                    'Z (raw)',
                    'Range (raw)',
                    'Curvature',
                    'Anisotropy',
                    'Planarity',
                    ]

    output_images = [density_image,
                     intensity_image_adjusted,
                     z_image_adjusted,
                     range_image_adjusted,
                     intensity_image,
                     z_image,
                     range_image,
                     curvature_image,
                     anisotropy_image,
                     planarity_image,
                     ]

    output_images_dict = {image_name: image for image_name, image in zip(image_names, output_images)}

    # Add the True RGB image if available
    if dataset_name == 'SEMANTIC3D':
        output_images_dict['True-RGB'] = rgb_image

        # Add the segmentation mask if available
        if 'class_id' in df_filtered_ncolored.columns:
            seg_mask_raw = np.full(canvas_size, 255, dtype=np.int16)
            grouped_class_id = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False)['class_id'].agg(lambda x: x.value_counts().idxmax())
            y_indices_class = grouped_class_id.index.get_level_values(0)
            x_indices_class = grouped_class_id.index.get_level_values(1)
            seg_mask_raw[y_indices_class, x_indices_class] = grouped_class_id.values
            seg_mask_raw[seg_mask_raw == -1] = 18
            seg_mask_raw = seg_mask_raw.astype(np.uint8)
            output_images_dict['seg_mask_raw'] = seg_mask_raw
            seg_mask_merged = seg_mask_raw.copy()
            seg_mask_merged[seg_mask_merged == 18] = 17
            seg_mask_merged[seg_mask_merged == 255] = 17
            output_images_dict['seg_mask_merged'] = seg_mask_merged

    return df_filtered_ncolored, output_images_dict


def equirectangular_projection_normals(df_filtered_ncolored: pd.DataFrame,
                                   canvas_size: tuple[int, int]) -> np.ndarray:
    """
    Unwrap the point cloud to 2D images with x being azimuth angle, y being zenith angle, and pixel values
    colorized by the normal vectors.

    Parameters:
        df_filtered_ncolored (pd.DataFrame): A DataFrame containing the filtered and processed point cloud data with additional columns for pixel coordinates.
        canvas_size (tuple[int, int]): The size of the canvas for the output image, (height, width).
    """
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False)

    r_per_pixel = grouped['n_r'].mean()
    g_per_pixel = grouped['n_g'].mean()
    b_per_pixel = grouped['n_b'].mean()


    ## Map the computed values to the image arrays
    ## Create empty image canvases with proper resolution
    r_image = np.zeros(canvas_size, dtype=np.float32)
    g_image = np.zeros(canvas_size, dtype=np.float32)
    b_image = np.zeros(canvas_size, dtype=np.float32)
    
    pxpy_indices = np.array(r_per_pixel.index.tolist())
    r_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = r_per_pixel.values
    g_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = g_per_pixel.values
    b_image[pxpy_indices[:, 0], pxpy_indices[:, 1]] = b_per_pixel.values

    # Stack the single channel images to create an RGB image
    rgb_image = np.stack((r_image, g_image, b_image), axis=-1) # Shape: (540, 1440, 3)
   
    return rgb_image



def save_image_cube_and_meta(
    output_images_dict: Dict[str, np.ndarray],
    normals_rgb_image: np.ndarray,
    output_dir: Path,
    key_str: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Saves an image cube and its metadata.

    This function constructs an image cube by stacking different adjusted maps
    (Intensity, Z, Range) along with the normal vector RGB
    image and then saves it as a `.npy` file. Additionally, metadata related to
    the image processing steps and other relevant information is saved in a separate
    `.npy` file.

    Parameters:
    - output_images_dict: A dictionary mapping titles of processed maps to numpy arrays.
    - normals_rgb_image: A numpy array representing the RGB image of normal vectors.
    - output_dir: The directory where the output files will be saved.
    - key_str: A string to be included in the filenames for the saved files.

    Returns:
    - None
    """

    # Titles for the image channels
    save_titles = [
        'Intensity (raw)', 
        'Z (raw)',
        'Range (raw)', 
        'Intensity (adjusted)',
        'Z-Inv (adjusted)',
        'Range (adjusted)',
        'Curvature',
        'Anisotropy',
        'Planarity',
    ]

    
    # Dynamically fetch the adjusted maps
    collected_maps = []
    for title in save_titles:
        try:
            feature_map = output_images_dict[title]
            # Examine the shape of the adjusted map
            if feature_map.shape != normals_rgb_image.shape[:2]:
                raise ValueError(f"Shape mismatch: {title} shape: {feature_map.shape}, normals_rgb_image shape: {normals_rgb_image.shape[:2]}")
            collected_maps.append(output_images_dict[title])
        except KeyError as e:
            print(f"Missing key: {e}")
            raise KeyError(f"Key {e} not found in the output_images_dict.")

    raw_maps = np.stack(collected_maps[:3], axis=-1) # shape: (H, W, 3)
    
    adjusted_maps = np.stack(collected_maps[3:], axis=-1) # shape: (H, W, N)
    pca_input_cube = np.concatenate([adjusted_maps, normals_rgb_image], axis=-1) # shape: (H, W, N+3)
    # Normalize the each channel of pca_input_cube to (0.0, 1.0)
    pca_input_cube_standardized = z_score_standardize(pca_input_cube)

    pcs, _ = compute_pca_components(pca_input_cube_standardized, n_components=3) # shape: (H, W, 3)
    mnf_components = compute_mnf(pca_input_cube_standardized, n_components=3)
    ica_components = compute_ica(pca_input_cube_standardized, n_components=3)
    extended_image_cube = np.concatenate([raw_maps, pca_input_cube_standardized, pcs, mnf_components, ica_components], axis=-1) # shape: (H, W, 18)

    if 'True-RGB' in output_images_dict:
        save_titles = ['True-R', 'True-G', 'True-B'] + save_titles
        true_rgb_image = output_images_dict['True-RGB'] if 'True-RGB' in output_images_dict else None
        if true_rgb_image is not None:
            extended_image_cube = np.concatenate([true_rgb_image, extended_image_cube], axis=-1)

    save_titles += ['Pseduo-Rn', 'Pseudo-Gn', 'Pseudo-Bn',
                    'PCA1', 'PCA2', 'PCA3', 
                    'MNF1', 'MNF2', 'MNF3', 
                    'ICA1', 'ICA2', 'ICA3']
    # Save the image cube
    image_cube_path = output_dir / f'{key_str}_image_cube.npy'
    np.save(image_cube_path, extended_image_cube)
    print(f"Image cube saved to {image_cube_path}, shape: {extended_image_cube.shape}")

    # Prepare metadata dynamically
    preprocess_meta = {}
    hist_visuals = {}
    for i, title in enumerate(save_titles):
        obs_image = extended_image_cube[:, :, i]
        histogram, _ = np.histogram(obs_image, bins=20)
        hist_visual = histogram_to_ascii(histogram, style="blocks")
        hist_visuals[title] = f"histogram_20_bins: {histogram.tolist()} | visual: {hist_visual}"
    preprocess_meta["histograms per channel"] = hist_visuals


    notes_per_channel = {}
    for i, title in enumerate(save_titles):
        if "True-" in title:
            notes_per_channel[title] = f"Ch{i+1} - {title} image."
        elif 'raw' in title:
            notes_per_channel[title] = f"Ch{i+1} - Raw data."
        elif 'adjusted' in title:
            notes_per_channel[title] = f"Ch{i+1} - {title} - first normalized all the values to (0.01, 1.0), then did histogram equalization on the valid pixels."
        elif "Pseduo-" in title:
            notes_per_channel[title] = f"Ch{i+1} - {title} - image colorized by mapping nx-ny-nz to (azimuth, zenith) and then to Hue(azi)-Saturation(0.5)-Value(zen) (HSV) color space and converted to RGB."
        elif "PCA" in title:
            notes_per_channel[title] = f"Ch{i+1} - PCA component {i+1-9} from intensity-range-Z-Rn-Gn-Bn."
        elif "MNF" in title:
            notes_per_channel[title] = f"Ch{i+1} - MNF component {i+1-12} from intensity-range-Z-Rn-Gn-Bn."
        elif "ICA" in title:
            notes_per_channel[title] = f"Ch{i+1} - ICA component {i+1-15} from intensity-range-Z-Rn-Gn-Bn."
        

    preprocess_meta["Notes per channel"] = notes_per_channel
    

    metadata = {
        'channel_names': save_titles,
        'shape': list(extended_image_cube.shape),
        'dtype': str(extended_image_cube.dtype),
        'key_str': key_str,
        'preprocess_meta': preprocess_meta
    }

    metadata_path = output_dir / f'{key_str}_image_cube_meta.yaml'
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)
    print(f"Metadata saved to {metadata_path}")

    return extended_image_cube, metadata



def load_image_cube_and_meta(image_cube_path: Path) -> Dict[str, Any]:
    """
    Loads an image cube and its metadata from saved .npy files.

    Parameters:
    - image_cube_path: The path to the saved image cube file.

    Returns:
    - A dictionary containing the image cube and metadata.
    """
    
    # Load the image cube (8-channel data)
    image_cube = np.load(image_cube_path)
    print(f"Image cube loaded from {image_cube_path}, shape: {image_cube.shape}")
    
    # Load the metadata
    metadata_path = image_cube_path.parent / f"{image_cube_path.stem}_meta.yaml"
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    print(f"Metadata loaded from {metadata_path}")
    return image_cube, metadata



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


def create_pseudo_rgb_image(img_ch1: np.ndarray, 
                            img_ch2: np.ndarray, 
                            img_ch3: np.ndarray, 
                            figure_title: str = '',
                            output_dir:Path=None,
                            saveflag:bool=False,
                            visualize: bool = True) -> np.ndarray:
    """
    Combine intensity, range, and density images into a pseudo-RGB image.

    Args:
        img_ch1 (np.ndarray): First image channel (e.g., intensity).
        img_ch2 (np.ndarray): Second image channel (e.g., range).
        img_ch3 (np.ndarray): Third image channel (e.g., Z-Inv).
        figure_title (str): Title for the figure.
        output_dir (Path): Directory to save the output image.
        saveflag (bool): If True, save the image.
        visualize (bool): If True, display the image.
    """
    # Normalize each channel before stacking them
    image_list = [img_ch1, img_ch2, img_ch3]
    pseudo_rgb_image = normalize_and_stack_images(image_list, method="global")
    display_unwrapped_rgb_image(pseudo_rgb_image, 
                            figure_title, 
                            output_dir, 
                            saveflag,
                            visualize
                            )

    return pseudo_rgb_image


def generate_2D_projection_images(output_dir: Path,
                                    canvas_size: tuple[int, int], 
                                    angular_res: tuple[int, int], 
                                    dataset_name: str = 'MANGROVE',
                                    color_map: Dict[str, tuple] = None,
                                    saveflag: bool = False,
                                    visualize: bool = True,
                                    extra_maps: bool = True,
                                    show_single_band: bool = True,
                                    show_pseudo_rgb: bool = True,
                                    show_pca: bool = True,
                                    v_fov: tuple[float, float] = (0, 90),
                                    h_fov: tuple[float, float] = (0, 360)
                                    ) -> None:
    """
    Generate 2D projection images from point cloud data.
    Args:
        output_dir (Path): Directory containing the point cloud data.
        canvas_size (tuple[int, int]): Size of the canvas for the output images.
        angular_res (tuple[int, int]): Angular resolution for the projection.
        saveflag (bool): If True, save the images.
        visualize (bool): If True, display the images.
        detailed_outputs (bool): If True, generate detailed outputs.
        show_single_band (bool): If True, show single band images.
        show_pseudo_rgb (bool): If True, show pseudo RGB images.
        show_pca (bool): If True, show PCA components.
        v_fov (tuple[float, float]): Vertical field of view.
        h_fov (tuple[float, float]): Horizontal field of view.
    """


    input_file_stem = output_dir.parent.name
    pcd_dir = output_dir / 'pcd'
    img_out_dir = output_dir / 'img'
    
    create_dir_if_not_exists(img_out_dir, ask_user=False)
    filename = next(pcd_dir.glob("*_ncr_0.02*"), None)
    if filename is None:
        raise FileNotFoundError("No file containing '_filtered_' found in output_dir.")

    titles = ['Intensity (adjusted)', 
            'Z-Inv (adjusted)',
            'Range (adjusted)', 
            'Intensity (raw)', 
            'Z (raw)',
            'Range (raw)', 
            'Curvature',
            'Anisotropy',
            'Planarity',
            ]
    
    key_str = input_file_stem.split('_')[0] + '_' + input_file_stem.split('_')[-1]
    df_filtered, output_images_dict = equirectangular_projection_multi(filename, canvas_size, angular_res, dataset_name)
    normals_rgb_image = equirectangular_projection_normals(df_filtered, canvas_size)
    image_cube, _ = save_image_cube_and_meta(output_images_dict, 
                            normals_rgb_image, 
                            img_out_dir, 
                            key_str, 
                            )
    # Plot confusion matrix of the pca_cube
    output_stem = f'{key_str}_image_cube'
    band_names = ['Intensity', 'Z-Inv', 'Range', 'Curvature', 'Anisotropy', 'Planarity']
    band_keys = [name + ' (adjusted)' for name in band_names[:3]] + ['Curvature', 'Anisotropy', 'Planarity']
    corr_matrix_input = np.stack([output_images_dict[name] for name in band_keys], axis=-1)
    corr_matrix_input = np.concatenate([corr_matrix_input, normals_rgb_image], axis=-1)
    corr_matrix = compute_band_correlation(corr_matrix_input)
    band_names += ['Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn']
    plot_correlation_matrix(corr_matrix, band_names = band_names, output_dir=img_out_dir, output_stem=output_stem)

    if dataset_name == 'SEMANTIC3D' and "True-RGB" in output_images_dict:
        true_rgb_image = output_images_dict['True-RGB']
        display_unwrapped_rgb_image(true_rgb_image, 
                                    figure_title=f'True_RGB_{key_str}', 
                                    saveflag=saveflag, 
                                    output_dir=img_out_dir,
                                    visualize=visualize,
                                    canvas_size=canvas_size,
                                    v_fov = v_fov,
                                    h_fov = h_fov
                                    )
    # Generate seg-mask from class_id
    if dataset_name == 'SEMANTIC3D' and 'class_id' in df_filtered.columns:
        seg_mask_merged = output_images_dict['seg_mask_merged']
        seg_map_merged_title = f'seg_map_merged_{output_dir.name}'
        seg_mask_merged_out = Image.fromarray(seg_mask_merged)
        seg_mask_merged_out.save(img_out_dir / f"{seg_map_merged_title}_mask.png")

        seg_mask_raw = output_images_dict['seg_mask_raw']
        seg_map_raw_title = f'seg_map_raw_{output_dir.name}'
        seg_mask_raw_out = Image.fromarray(seg_mask_raw)
        seg_mask_raw_out.save(img_out_dir / f"{seg_map_raw_title}_mask.png")
        display_single_band_img_wt_discrete_values(seg_mask_merged,
                                                title=seg_map_merged_title, 
                                                num_unique_values=18,
                                                cb_label='Class ID',
                                                output_dir=img_out_dir, 
                                                saveflag=saveflag,
                                                visualize=visualize,
                                                color_map=color_map
                                                )

    if extra_maps:
        density_image = output_images_dict['Density']
        display_single_band_img_wt_discrete_values(density_image, 
                                                title='Point Density Map', 
                                                cb_label='Point Density',
                                                output_dir=img_out_dir, 
                                                saveflag=saveflag,
                                                visualize=visualize
                                                )
        if show_single_band:
            # Display and save the adjusted intensity and range images
            display_images = [output_images_dict[title] for title in titles]
            display_unwrapped_single_band_images(display_images, 
                                                titles=titles,
                                                key_str=key_str,
                                                output_dir=img_out_dir,
                                                saveflag=saveflag,
                                                visualize=visualize,
                                                canvas_size=canvas_size,
                                                v_fov = v_fov,
                                                h_fov = h_fov
                                                )
        if show_pseudo_rgb:
            # Display the Pseudo-RGB image from normals.
            display_unwrapped_rgb_image(normals_rgb_image, 
                                        figure_title=f'HSV_colorized_map_from_normals_{key_str}', 
                                        saveflag=saveflag, 
                                        output_dir=img_out_dir,
                                        visualize=visualize,
                                        canvas_size=canvas_size,
                                        v_fov = v_fov,
                                        h_fov = h_fov
                                        )
            # Create pseudo-RGB images from various combinations of intensity, range, and Z-Inv.
            feat_strs = ['Intensity', 'Z-Inv', 'Range', 'Curvature', 'Anisotropy', 'Planarity']
            feature_maps = [output_images_dict[key + ' (adjusted)'] for key in feat_strs[:3]] + \
                            [output_images_dict[key] for key in feat_strs[3:]]
            # Create shuffle orders for the feature maps using permutations and randomly select 3 out of 6 features
            from itertools import permutations
            from random import sample
            from random import seed
            seed(617)  # For reproducibility
            shuffle_orders = list(permutations(range(len(feature_maps)), 3))
            shuffle_orders = sample(shuffle_orders, min(10, len(shuffle_orders)))
            figure_titles = [f'Pseudo-RGB_{feat_strs[shuffle_order[0]]}-{feat_strs[shuffle_order[1]]}-{feat_strs[shuffle_order[2]]}_{key_str}' for shuffle_order in shuffle_orders]
            for (shuffle_order, figure_title) in zip(shuffle_orders, figure_titles):
                create_pseudo_rgb_image(feature_maps[shuffle_order[0]], 
                                        feature_maps[shuffle_order[1]], 
                                        feature_maps[shuffle_order[2]], 
                                        figure_title=figure_title, 
                                        output_dir=img_out_dir, 
                                        saveflag=saveflag, 
                                        visualize=visualize)
            
        
        if show_pca:
            # Display PCA, MNF, and ICA components
            pcs = image_cube[:, :, -9:-6]
            mnf_components = image_cube[:, :, -6:-3]
            ica_components = image_cube[:, :, -3:]
            for components, name in zip([pcs, mnf_components, ica_components], ['PCA', 'MNF', 'ICA']):
                out_file = f"{output_stem}_{name}"
                plot_pca_components(components, img_out_dir, output_stem=out_file)
                plot_rgb_permutations(components, img_out_dir, output_stem=out_file)
def main():
    params = CONFIG['spherical_projection']
    color_map = CONFIG['global']['color_map']
    save_extra_maps = params['save_extra_maps']
    visualize = params['visualize']
    extra_maps = params['extra_maps']
    v_fov = CONFIG['global']['v_fov']
    h_fov = CONFIG['global']['h_fov']
    canvas_width = int((h_fov[1]-h_fov[0]) / CONFIG['global']['h_ang_res_deg'])
    canvas_height = int((v_fov[1]-v_fov[0]) / CONFIG['global']['v_ang_res_deg'])
    canvas_size = (canvas_height, canvas_width)
    angular_res = (CONFIG['global']['v_ang_res_deg'], CONFIG['global']['h_ang_res_deg'])
    dataset_name = CONFIG['global']['dataset']
    output_dir_ls = CONFIG['global']['output_dir_ls']
    show_single_band = params['show_single_band']
    show_pseudo_rgb = params['show_pseudo_rgb']
    show_pca = params['show_pca']
    for output_dir in output_dir_ls:
        generate_2D_projection_images(output_dir=output_dir,
                                    canvas_size=canvas_size, 
                                    angular_res=angular_res, 
                                    dataset_name=dataset_name,
                                    color_map=color_map,
                                    saveflag=save_extra_maps,
                                    visualize=visualize,
                                    extra_maps=extra_maps,
                                    show_single_band=show_single_band,
                                    show_pseudo_rgb=show_pseudo_rgb,
                                    show_pca=show_pca,
                                    v_fov=v_fov,
                                    h_fov=h_fov
                                        )

if __name__ == "__main__":
    main()