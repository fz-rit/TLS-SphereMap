import numpy as np
import yaml
import xarray as xr
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image
from itertools import permutations
from random import sample, seed
import pandas as pd

from tools.plot_tools import (
    plot_correlation_matrix, plot_pca_components, plot_components_permutations,
    display_unwrapped_single_band_images, display_unwrapped_rgb_image, 
    display_single_band_img_wt_discrete_values, histogram_to_ascii
)
from tools.pca_helper import (
    compute_band_correlation, compute_pca_components, compute_mnf, 
    compute_ica, z_score_standardize
)


def get_channel_data(projection_xr: xr.DataArray, channel_name: str) -> np.ndarray:
    """Extract a specific channel from the projection DataArray.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        channel_name: Name of the channel to extract
        
    Returns:
        2D numpy array for the specified channel
        
    Raises:
        KeyError: If channel_name is not found
    """
    if channel_name not in projection_xr.coords['channel'].values:
        available_channels = list(projection_xr.coords['channel'].values)
        raise KeyError(f"Channel '{channel_name}' not found. Available channels: {available_channels}")
    
    return projection_xr.sel(channel=channel_name).values


def get_rgb_channels(projection_xr: xr.DataArray, rgb_type: str = 'true') -> np.ndarray:
    """Extract RGB channels and combine into a 3D array.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        rgb_type: Type of RGB ('true' for True-RGB)
        
    Returns:
        3D numpy array of shape (height, width, 3) for RGB visualization
        
    Raises:
        ValueError: If RGB channels are not available
    """
    if rgb_type == 'true':
        channel_names = ['true_r', 'true_g', 'true_b']
    else:
        raise ValueError(f"Unsupported rgb_type: {rgb_type}")
    
    # Check if all required channels exist
    available_channels = list(projection_xr.coords['channel'].values)
    missing_channels = [ch for ch in channel_names if ch not in available_channels]
    
    if missing_channels:
        raise ValueError(f"Missing RGB channels: {missing_channels}")
    
    # Extract and stack RGB channels
    rgb_data = []
    for channel in channel_names:
        rgb_data.append(projection_xr.sel(channel=channel).values)
    
    return np.stack(rgb_data, axis=-1)


def list_available_channels(projection_xr: xr.DataArray) -> List[str]:
    """List all available channels in the projection DataArray.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        
    Returns:
        List of available channel names
    """
    return list(projection_xr.coords['channel'].values)


def create_channel_subset(projection_xr: xr.DataArray, channels: List[str]) -> xr.DataArray:
    """Create a subset DataArray with only specified channels.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        channels: List of channel names to include
        
    Returns:
        New DataArray with only the specified channels
        
    Raises:
        KeyError: If any specified channel is not found
    """
    available_channels = list(projection_xr.coords['channel'].values)
    missing_channels = [ch for ch in channels if ch not in available_channels]
    
    if missing_channels:
        raise KeyError(f"Missing channels: {missing_channels}. Available: {available_channels}")
    
    return projection_xr.sel(channel=channels)

def save_image_cube_and_meta(
    projection_xr: xr.DataArray,
    normals_rgb_image: np.ndarray,
    output_dir: Path,
    key_str: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Save multi-channel image cube with metadata.
    
    Creates an extended image cube by stacking various processed maps,
    computing PCA/MNF/ICA components, and saving with comprehensive metadata.

    Args:
        projection_xr: Multi-channel projection DataArray
        normals_rgb_image: RGB image from normal vectors
        output_dir: Directory for saving outputs
        key_str: Identifier string for filenames

    Returns:
        Tuple containing the extended image cube and metadata dictionary

    Raises:
        KeyError: If required channels are missing from projection_xr
        ValueError: If image shapes don't match
    """
    # Define channel order for saving
    save_channels = [
        'intensity_raw', 'z_raw', 'range_raw', 
        'intensity_adjusted', 'z_adjusted', 'range_adjusted',
        'curvature', 'anisotropy', 'planarity',
    ]
    
    # save_titles = [
    #     'Intensity (raw)', 'Z (raw)', 'Range (raw)', 
    #     'Intensity (adjusted)', 'Z-Inv (adjusted)', 'Range (adjusted)',
    #     'Curvature', 'Anisotropy', 'Planarity',
    # ]

    # Validate and collect maps
    collected_maps = []
    expected_shape = normals_rgb_image.shape[:2]
    
    for channel in save_channels:
        if channel not in projection_xr.coords['channel'].values:
            raise KeyError(f"Required channel '{channel}' not found in projection_xr")
        
        feature_map = get_channel_data(projection_xr, channel)
        if feature_map.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{channel}': expected {expected_shape}, "
                f"got {feature_map.shape}"
            )
        collected_maps.append(feature_map)

    # Organize image components
    raw_maps = np.stack(collected_maps[:3], axis=-1)
    feature_maps = np.stack(collected_maps[3:], axis=-1)
    
    # Create standardized input for dimensionality reduction
    pca_input_cube = np.concatenate([feature_maps, normals_rgb_image], axis=-1)
    pca_input_standardized = z_score_standardize(pca_input_cube)

    # Compute dimensionality reduction components
    pcs, _ = compute_pca_components(pca_input_standardized, n_components=3)
    mnf_components = compute_mnf(pca_input_standardized, n_components=3)
    ica_components = compute_ica(pca_input_standardized, n_components=3)

    # Build extended image cube
    extended_cube = np.concatenate([
        raw_maps, pca_input_cube, pcs, mnf_components, ica_components
    ], axis=-1)

    # Handle optional True-RGB data
    if 'true_r' in projection_xr.coords['channel'].values:
        true_rgb = get_rgb_channels(projection_xr, 'true')
        extended_cube = np.concatenate([true_rgb, extended_cube], axis=-1)
        save_channels = ['True-R', 'True-G', 'True-B'] + save_channels

    # Add component labels
    save_channels.extend([
        'Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn',
        'PCA1', 'PCA2', 'PCA3', 
        'MNF1', 'MNF2', 'MNF3',
        'ICA1', 'ICA2', 'ICA3'
    ])

    # Save image cube
    cube_path = output_dir / f'{key_str}_image_cube.npy'
    np.save(cube_path, extended_cube)
    print(f"Image cube saved: {cube_path}, shape: {extended_cube.shape}")

    # Generate metadata
    metadata = _generate_cube_metadata(extended_cube, save_channels, key_str)
    
    # Save metadata
    meta_path = output_dir / f'{key_str}_image_cube_meta.yaml'
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)
    print(f"Metadata saved: {meta_path}")

    return extended_cube, metadata


def _generate_cube_metadata(
    image_cube: np.ndarray, 
    channel_names: List[str], 
    key_str: str
) -> Dict[str, Any]:
    """Generate comprehensive metadata for image cube.
    
    Args:
        image_cube: Multi-channel image array
        channel_names: List of channel names
        key_str: Identifier string
        
    Returns:
        Dictionary containing metadata
    """
    # Generate histograms and descriptions for each channel
    hist_visuals = {}
    channel_notes = {}
    
    for i, title in enumerate(channel_names):
        channel_data = image_cube[:, :, i]
        histogram, bin_edges = np.histogram(channel_data, bins=20)
        histogram_20_bins_ls = [list(pair) for pair in zip(np.round(bin_edges[:-1], decimals=4).tolist(), histogram.tolist())]
        hist_visuals[title] = {
            'histogram_20_bins': histogram_20_bins_ls,
            'visual': histogram_to_ascii(histogram, style="blocks")
        }

        
        # Generate channel descriptions
        channel_notes[title] = _get_channel_description(title, i + 1)

    return {
        'channel_names': channel_names,
        'shape': list(image_cube.shape),
        'dtype': str(image_cube.dtype),
        'key_str': key_str,
        'preprocess_meta': {
            'histograms_per_channel': hist_visuals,
            'notes_per_channel': channel_notes
        }
    }


def _get_channel_description(title: str, channel_num: int) -> str:
    """Generate description for a specific channel.
    
    Args:
        title: Channel title
        channel_num: Channel number
        
    Returns:
        Description string for the channel
    """
    descriptions = {
        'True-': f"Ch{channel_num} - {title} image from original RGB data.",
        'raw': f"Ch{channel_num} - Raw {title.split('(')[0].strip()} data.",
        'adjusted': f"Ch{channel_num} - {title} - normalized to (0.01, 1.0), then histogram equalization.",
        'Pseudo-': f"Ch{channel_num} - {title} - normal vectors mapped to HSV color space.",
        'PCA': f"Ch{channel_num} - PCA component from multi-spectral features.",
        'MNF': f"Ch{channel_num} - MNF component from multi-spectral features.",
        'ICA': f"Ch{channel_num} - ICA component from multi-spectral features.",
    }
    
    for key, desc in descriptions.items():
        if key in title:
            return desc
    
    return f"Ch{channel_num} - {title} feature map."


def load_image_cube_and_meta(image_cube_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load saved image cube and its metadata.

    Args:
        image_cube_path: Path to the saved image cube .npy file

    Returns:
        Tuple containing the image cube array and metadata dictionary
        
    Raises:
        FileNotFoundError: If cube or metadata files don't exist
    """
    if not image_cube_path.exists():
        raise FileNotFoundError(f"Image cube file not found: {image_cube_path}")
    
    # Load image cube
    image_cube = np.load(image_cube_path)
    print(f"Image cube loaded: {image_cube_path}, shape: {image_cube.shape}")
    
    # Load metadata
    metadata_path = image_cube_path.parent / f"{image_cube_path.stem}_meta.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    print(f"Metadata loaded: {metadata_path}")
    
    return image_cube, metadata


def normalize_and_stack_images(image_list: List[np.ndarray]) -> np.ndarray:
    """Normalize each image to [0, 1] independently and stack as channels.

    Args:
        image_list: List of 2D image arrays (H, W) to normalize and stack

    Returns:
        A 3D array (H, W, C) with each channel normalized to [0, 1]
    """
    normalized_images = []
    for img in image_list:
        min_val, max_val = img.min(), img.max()
        norm_img = (img - min_val) / (max_val - min_val + 1e-8)
        normalized_images.append(norm_img)

    stacked_image = np.stack(normalized_images, axis=-1)
    return stacked_image


def create_pseudo_rgb_image(
    img_ch1: np.ndarray, 
    img_ch2: np.ndarray, 
    img_ch3: np.ndarray, 
    figure_title: str = '',
    output_dir: Optional[Path] = None,
    saveflag: bool = False,
    visualize: bool = True
) -> np.ndarray:
    """Create pseudo-RGB image from three single-channel images.

    Args:
        img_ch1: First channel (assigned to R)
        img_ch2: Second channel (assigned to G)  
        img_ch3: Third channel (assigned to B)
        figure_title: Title for the generated figure
        output_dir: Directory to save the image (if saveflag=True)
        saveflag: Whether to save the image
        visualize: Whether to display the image

    Returns:
        Normalized pseudo-RGB image array
    """
    image_list = [img_ch1, img_ch2, img_ch3]
    pseudo_rgb_image = normalize_and_stack_images(image_list)
    
    display_unwrapped_rgb_image(
        pseudo_rgb_image, 
        figure_title, 
        output_dir, 
        saveflag,
        visualize
    )

    return pseudo_rgb_image


def generate_correlation_matrix(
    projection_xr: xr.DataArray,
    normals_rgb_image: np.ndarray,
    img_out_dir: Path,
    key_str: str
) -> None:
    """Generate and save correlation matrix plot.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        normals_rgb_image: RGB image from normal vectors
        img_out_dir: Output directory for plots
        key_str: Identifier string for filenames
    """
    # Define channels to include in correlation analysis
    correlation_channels = [
        'intensity_adjusted', 'z_adjusted', 'range_adjusted',
        'curvature', 'anisotropy', 'planarity'
    ]
    
    band_names = ['Intensity', 'Z-Inv', 'Range', 'Curvature', 'Anisotropy', 'Planarity']
    
    # Extract channels and stack
    corr_images = []
    for channel in correlation_channels:
        if channel in projection_xr.coords['channel'].values:
            corr_images.append(get_channel_data(projection_xr, channel))
    
    corr_input = np.stack(corr_images, axis=-1)
    corr_input = np.concatenate([corr_input, normals_rgb_image], axis=-1)
    
    corr_matrix = compute_band_correlation(corr_input)
    band_names.extend(['Pseudo-Rn', 'Pseudo-Gn', 'Pseudo-Bn'])
    
    plot_correlation_matrix(
        corr_matrix, 
        band_names=band_names, 
        output_dir=img_out_dir, 
        output_stem=f'{key_str}_image_cube'
    )


def generate_semantic3d_outputs(
    projection_xr: xr.DataArray,
    df_filtered: pd.DataFrame,
    img_out_dir: Path,
    key_str: str,
    color_map: Optional[Dict[str, tuple]],
    saveflag: bool,
    visualize: bool,
    canvas_size: Tuple[int, int],
    v_fov: Tuple[float, float],
    h_fov: Tuple[float, float]
) -> None:
    """Generate SEMANTIC3D-specific visualizations.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        df_filtered: Original DataFrame with point cloud data
        img_out_dir: Output directory for images
        key_str: Identifier string for filenames
        color_map: Color mapping for segmentation visualization
        saveflag: Whether to save generated images
        visualize: Whether to display images
        canvas_size: Output image dimensions
        v_fov: Vertical field of view range
        h_fov: Horizontal field of view range
    """
    # True RGB image
    if 'true_r' in projection_xr.coords['channel'].values:
        true_rgb = get_rgb_channels(projection_xr, 'true')
        display_unwrapped_rgb_image(
            true_rgb, 
            figure_title=f'True_RGB_{key_str}', 
            saveflag=saveflag, 
            output_dir=img_out_dir,
            visualize=visualize,
            canvas_size=canvas_size,
            v_fov=v_fov,
            h_fov=h_fov
        )

    # Segmentation masks
    if 'class_id' in df_filtered.columns:
        for mask_type in ['merged', 'raw']:
            mask_channel = f'seg_mask_{mask_type}'
            if mask_channel in projection_xr.coords['channel'].values:
                seg_mask = get_channel_data(projection_xr, mask_channel).astype(np.uint8)
                title = f'seg_map_{mask_type}_{key_str}'
                
                # Save mask as PNG
                Image.fromarray(seg_mask).save(img_out_dir / f"{title}_mask.png")
                
                if mask_type == 'merged':
                    display_single_band_img_wt_discrete_values(
                        seg_mask,
                        title=title, 
                        num_unique_values=18,
                        cb_label='Class ID',
                        output_dir=img_out_dir, 
                        saveflag=saveflag,
                        visualize=visualize,
                        color_map=color_map
                    )


def generate_forest_semantic_outputs(
    projection_xr: xr.DataArray,
    df_filtered: pd.DataFrame,
    img_out_dir: Path,
    key_str: str,
    color_map: Optional[Dict[str, tuple]],
    saveflag: bool,
    visualize: bool,
    # canvas_size: Tuple[int, int],
    # v_fov: Tuple[float, float],
    # h_fov: Tuple[float, float]
) -> None:
    """Generate SEMANTIC3D-specific visualizations.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        df_filtered: Original DataFrame with point cloud data
        img_out_dir: Output directory for images
        key_str: Identifier string for filenames
        color_map: Color mapping for segmentation visualization
        saveflag: Whether to save generated images
        visualize: Whether to display images
        canvas_size: Output image dimensions
        v_fov: Vertical field of view range
        h_fov: Horizontal field of view range
    """

    # Segmentation masks
    if 'Classification' in df_filtered.columns:
            mask_channel = f'seg_mask'
            if mask_channel in projection_xr.coords['channel'].values:
                seg_mask = get_channel_data(projection_xr, mask_channel).astype(np.uint8)
                title = f'seg_map_{key_str}'
                
                # Save mask as PNG
                Image.fromarray(seg_mask).save(img_out_dir / f"{title}_mask.png")
                display_single_band_img_wt_discrete_values(
                        seg_mask,
                        title=title, 
                        num_unique_values=7,
                        cb_label='Class ID',
                        output_dir=img_out_dir, 
                        saveflag=saveflag,
                        visualize=visualize,
                        color_map=color_map
                    )

def generate_extra_visualizations(
    projection_xr: xr.DataArray,
    normals_rgb_image: np.ndarray,
    image_cube: np.ndarray,
    img_out_dir: Path,
    key_str: str,
    saveflag: bool,
    visualize: bool,
    show_single_band: bool,
    show_pseudo_rgb: bool,
    show_pca: bool,
    canvas_size: Tuple[int, int],
    v_fov: Tuple[float, float],
    h_fov: Tuple[float, float]
) -> None:
    """Generate additional visualization outputs.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        normals_rgb_image: RGB image from normal vectors
        image_cube: Extended image cube with PCA/MNF/ICA components
        img_out_dir: Output directory for images
        key_str: Identifier string for filenames
        saveflag: Whether to save generated images
        visualize: Whether to display images
        show_single_band: Whether to show individual band images
        show_pseudo_rgb: Whether to generate pseudo-RGB combinations
        show_pca: Whether to show PCA/MNF/ICA components
        canvas_size: Output image dimensions
        v_fov: Vertical field of view range
        h_fov: Horizontal field of view range
    """
    # Point density map
    if 'density' in projection_xr.coords['channel'].values:
        density_map = get_channel_data(projection_xr, 'density')
        display_single_band_img_wt_discrete_values(
            density_map, 
            title='Point Density Map', 
            cb_label='Point Density',
            output_dir=img_out_dir, 
            saveflag=saveflag,
            visualize=visualize
        )

    if show_single_band:
        # Define channels to display
        display_channels = [
            'intensity_adjusted', 'z_adjusted', 'range_adjusted', 
            'intensity_raw', 'z_raw', 'range_raw', 
            'curvature', 'anisotropy', 'planarity'
        ]
        
        display_titles = [
            'Intensity (adjusted)', 'Z-Inv (adjusted)', 'Range (adjusted)', 
            'Intensity (raw)', 'Z (raw)', 'Range (raw)', 
            'Curvature', 'Anisotropy', 'Planarity'
        ]
        
        # Extract available channels
        available_images = []
        available_titles = []
        
        for channel, title in zip(display_channels, display_titles):
            if channel in projection_xr.coords['channel'].values:
                available_images.append(get_channel_data(projection_xr, channel))
                available_titles.append(title)
        
        if available_images:
            display_unwrapped_single_band_images(
                available_images, 
                titles=available_titles,
                key_str=key_str,
                output_dir=img_out_dir,
                saveflag=saveflag,
                visualize=visualize,
                canvas_size=canvas_size,
                v_fov=v_fov,
                h_fov=h_fov
            )

    if show_pseudo_rgb:
        # Normal-based RGB
        display_unwrapped_rgb_image(
            normals_rgb_image, 
            figure_title=f'HSV_colorized_map_from_normals_{key_str}', 
            saveflag=saveflag, 
            output_dir=img_out_dir,
            visualize=visualize,
            canvas_size=canvas_size,
            v_fov=v_fov,
            h_fov=h_fov
        )

        # Generate pseudo-RGB combinations
        generate_pseudo_rgb_combinations(
            projection_xr, key_str, img_out_dir, saveflag, visualize
        )

    if show_pca:
        # Display PCA, MNF, and ICA components
        components_data = [
            (image_cube[:, :, -9:-6], 'PCA'),
            (image_cube[:, :, -6:-3], 'MNF'), 
            (image_cube[:, :, -3:], 'ICA')
        ]
        
        for components, name in components_data:
            output_stem = f"{key_str}_image_cube_{name}"
            plot_pca_components(components, img_out_dir, output_stem=output_stem)
            plot_components_permutations(components, img_out_dir, output_stem=output_stem)


def generate_pseudo_rgb_combinations(
    projection_xr: xr.DataArray,
    key_str: str,
    img_out_dir: Path,
    saveflag: bool,
    visualize: bool
) -> None:
    """Generate pseudo-RGB images from various feature combinations.
    
    Args:
        projection_xr: Multi-channel projection DataArray
        key_str: Identifier string for filenames
        img_out_dir: Output directory for images
        saveflag: Whether to save generated images
        visualize: Whether to display images
    """
    # Define feature channels and their display names
    feature_channels = [
        'intensity_adjusted', 'z_adjusted', 'range_adjusted',
        'curvature', 'anisotropy', 'planarity'
    ]
    
    feat_names = ['Intensity', 'Z-Inv', 'Range', 'Curvature', 'Anisotropy', 'Planarity']
    
    # Extract available feature maps
    available_maps = []
    available_names = []
    
    for channel, name in zip(feature_channels, feat_names):
        if channel in projection_xr.coords['channel'].values:
            available_maps.append(get_channel_data(projection_xr, channel))
            available_names.append(name)
    
    if len(available_maps) < 3:
        print(f"Warning: Only {len(available_maps)} feature maps available, need at least 3 for RGB combinations")
        return

    # Generate random combinations
    seed(617)
    all_combinations = list(permutations(range(len(available_maps)), 3))
    selected_combinations = sample(all_combinations, min(3, len(all_combinations)))
    selected_combinations += [
        (0, 1, 2),  
        (3, 4, 5),  
    ]
    for combo in selected_combinations:
        feature_combo = [available_names[i] for i in combo]
        figure_title = f"Pseudo-RGB_{'_'.join(feature_combo)}_{key_str}"
        
        create_pseudo_rgb_image(
            available_maps[combo[0]], 
            available_maps[combo[1]], 
            available_maps[combo[2]], 
            figure_title=figure_title, 
            output_dir=img_out_dir, 
            saveflag=saveflag, 
            visualize=visualize
        )