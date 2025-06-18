import pandas as pd
import numpy as np
from tools.hdr_adjustments import contrast_enhancement
from tools.preprocess_point_cloud import map_angle_to_pixel
from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import xarray as xr
from tools.norm_to_hsv import attach_normal_color_to_df
from tools.config_loader import CONFIG
from tools.pcd_utils import create_dir_if_not_exists

from tools.spherical_projection_helper import (
    save_image_cube_and_meta, 
    generate_correlation_matrix, 
    generate_semantic3d_outputs,
    generate_extra_visualizations, 
)

pd.options.mode.chained_assignment = None


def load_and_preprocess_point_cloud(
    filename: Union[str, Path], 
    canvas_size: Tuple[int, int],
    angular_res: Tuple[int, int]
) -> pd.DataFrame:
    """Load and preprocess point cloud data.
    
    Performs the following preprocessing steps:
    1. Attach pixel coordinates for azimuth and elevation angles
    2. Attach HSV color values for normal vectors
    3. Shift Z values to start from 0 and make them negative
    4. Normalize Intensity and Z columns to range (0.01, 1.0)
    5. Validate data integrity (no NaN or negative values)

    Args:
        filename: Path to the point cloud CSV file
        canvas_size: Output image dimensions (height, width)
        angular_res: Angular resolution (vertical, horizontal) in degrees

    Returns:
        Preprocessed DataFrame with pixel coordinates and normalized values

    Raises:
        AssertionError: If data contains NaN or negative values in critical columns
    """
    df_filtered = pd.read_csv(filename, sep=',')
    
    # Map angles to pixel coordinates
    azimuth, elevation = df_filtered['azimuth'], df_filtered['elevation']
    x_pix, y_pix = map_angle_to_pixel(azimuth, elevation, canvas_size, angular_res)
    df_filtered['x_pix'] = x_pix
    df_filtered['y_pix'] = y_pix

    # Attach HSV colors from normal vectors
    df_filtered_ncolored = attach_normal_color_to_df(df_filtered)
    
    # Shift and invert Z values to start from 0
    z_min = df_filtered_ncolored['Z'].min()
    df_filtered_ncolored['Z'] = -(df_filtered_ncolored['Z'] - z_min)
    print("Z values shifted and inverted to start from 0.")

    # Normalize critical columns to (0.01, 1.0) range
    for col_name in ['Intensity', 'Z']:
        col_min = df_filtered_ncolored[col_name].min()
        col_max = df_filtered_ncolored[col_name].max()
        df_filtered_ncolored[col_name] = (
            0.01 + 0.99 * (df_filtered_ncolored[col_name] - col_min) / (col_max - col_min)
        )
        print(f"{col_name} normalized to (0.01, 1.0).")

    # Validate data integrity
    critical_columns = ['Intensity', 'azimuth', 'zenith', 'rangemeter']
    for col_name in critical_columns:
        assert df_filtered_ncolored[col_name].isnull().sum() == 0, \
            f"Column {col_name} contains NaN values"
        assert (df_filtered_ncolored[col_name] < 0).sum() == 0, \
            f"Column {col_name} contains negative values"

    return df_filtered_ncolored


def equirectangular_projection_multi(
    filename: str,
    canvas_size: Tuple[int, int],
    angular_res: Tuple[int, int],
    dataset_name: str = 'MANGROVE',
) -> Tuple[pd.DataFrame, xr.DataArray]:
    """Create multiple 2D equirectangular projections from point cloud data.
    
    Generates various scalar field projections including density, intensity, 
    range, geometric features, and optionally RGB and segmentation masks.

    Args:
        filename: Path to the point cloud CSV file
        canvas_size: Output image dimensions (height, width)
        angular_res: Angular resolution (vertical, horizontal) in degrees
        dataset_name: Dataset type ('MANGROVE' or 'SEMANTIC3D')

    Returns:
        Tuple containing:
        - Preprocessed DataFrame with pixel coordinates
        - xarray DataArray with multi-channel projection images

    Raises:
        ValueError: If dataset_name is not supported
    """
    df_filtered_ncolored = load_and_preprocess_point_cloud(
        filename, canvas_size=canvas_size, angular_res=angular_res
    )

    # Define aggregation strategy based on dataset
    base_agg = {
        'Intensity': 'mean',
        'Z': 'min',
        'rangemeter': 'mean',
        'curvature': 'mean',
        'anisotropy': 'mean',
        'planarity': 'mean',
    }

    if dataset_name == 'SEMANTIC3D':
        # Convert columns to float for SEMANTIC3D
        float_cols = ['rangemeter', 'r', 'g', 'b']
        for col in float_cols:
            if col in df_filtered_ncolored.columns:
                df_filtered_ncolored[col] = df_filtered_ncolored[col].astype(float)
        
        # Add RGB aggregation
        base_agg.update({
            'r': 'mean',
            'g': 'mean',
            'b': 'mean',
        })
    elif dataset_name != 'MANGROVE':
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: 'SEMANTIC3D', 'MANGROVE'"
        )

    # Group by pixel coordinates and aggregate
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).agg(base_agg)
    
    # Compute point density separately
    pts_per_pixel = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).size()

    # Define channel names and initialize data
    base_channels = [
        'intensity_raw', 'z_raw', 'range_raw',
        'intensity_adjusted', 'z_adjusted', 'range_adjusted',
        'curvature', 'anisotropy', 'planarity', 'density'
    ]
    
    # Add dataset-specific channels
    if dataset_name == 'SEMANTIC3D':
        base_channels.extend(['true_r', 'true_g', 'true_b'])
        if 'class_id' in df_filtered_ncolored.columns:
            base_channels.extend(['seg_mask_raw', 'seg_mask_merged'])

    # Create coordinate arrays
    y_coords = np.arange(canvas_size[0])
    x_coords = np.arange(canvas_size[1])
    
    # Initialize the multi-channel array
    projection_data = np.zeros((len(base_channels), canvas_size[0], canvas_size[1]), dtype=np.float32)
    
    # Extract pixel indices for vectorized assignment
    y_indices = grouped.index.get_level_values(0)
    x_indices = grouped.index.get_level_values(1)
    
    # Extract density indices
    density_y_indices = pts_per_pixel.index.get_level_values(0)
    density_x_indices = pts_per_pixel.index.get_level_values(1)

    # Create temporary arrays for processing
    image_arrays = {
        'intensity': np.zeros(canvas_size, dtype=np.float32),
        'range': np.zeros(canvas_size, dtype=np.float32),
        'z': np.zeros(canvas_size, dtype=np.float32),
        'density': np.zeros(canvas_size, dtype=np.float32),
        'curvature': np.zeros(canvas_size, dtype=np.float32),
        'anisotropy': np.zeros(canvas_size, dtype=np.float32),
        'planarity': np.zeros(canvas_size, dtype=np.float32),
    }

    # Populate temporary arrays
    image_arrays['intensity'][y_indices, x_indices] = grouped['Intensity'].values
    image_arrays['z'][y_indices, x_indices] = grouped['Z'].values
    image_arrays['range'][y_indices, x_indices] = grouped['rangemeter'].values
    image_arrays['density'][density_y_indices, density_x_indices] = pts_per_pixel.values
    image_arrays['curvature'][y_indices, x_indices] = grouped['curvature'].values
    image_arrays['anisotropy'][y_indices, x_indices] = grouped['anisotropy'].values
    image_arrays['planarity'][y_indices, x_indices] = grouped['planarity'].values

    # Apply contrast enhancement
    intensity_adjusted = contrast_enhancement(image_arrays['intensity'], stretch_percentile=0.1)
    z_adjusted = contrast_enhancement(image_arrays['z'], stretch_percentile=0)
    range_adjusted = contrast_enhancement(image_arrays['range'], stretch_percentile=0)

    # Normalize geometric features to [0, 1]
    for feature in ['curvature', 'anisotropy', 'planarity']:
        img = image_arrays[feature]
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        image_arrays[feature] = img_normalized

    # Populate the multi-channel array
    channel_mapping = {
        'intensity_raw': image_arrays['intensity'],
        'z_raw': image_arrays['z'],
        'range_raw': image_arrays['range'],
        'intensity_adjusted': intensity_adjusted,
        'z_adjusted': z_adjusted,
        'range_adjusted': range_adjusted,
        'curvature': image_arrays['curvature'],
        'anisotropy': image_arrays['anisotropy'],
        'planarity': image_arrays['planarity'],
        'density': image_arrays['density'],
    }

    # Add dataset-specific channels
    if dataset_name == 'SEMANTIC3D':
        # Create RGB channels
        rgb_r = np.zeros(canvas_size, dtype=np.float32)
        rgb_g = np.zeros(canvas_size, dtype=np.float32)
        rgb_b = np.zeros(canvas_size, dtype=np.float32)
        
        rgb_r[y_indices, x_indices] = grouped['r'].values
        rgb_g[y_indices, x_indices] = grouped['g'].values
        rgb_b[y_indices, x_indices] = grouped['b'].values
        
        channel_mapping.update({
            'true_r': rgb_r,
            'true_g': rgb_g,
            'true_b': rgb_b,
        })

        # Add segmentation masks if available
        if 'class_id' in df_filtered_ncolored.columns:
            seg_masks = _create_segmentation_masks(
                df_filtered_ncolored, canvas_size, grouped
            )
            channel_mapping.update({
                'seg_mask_raw': seg_masks['seg_mask_raw'].astype(np.float32),
                'seg_mask_merged': seg_masks['seg_mask_merged'].astype(np.float32),
            })

    # Fill the projection data array
    for i, channel in enumerate(base_channels):
        if channel in channel_mapping:
            projection_data[i] = channel_mapping[channel]

    # Create xarray DataArray with proper coordinates and metadata
    projection_xr = xr.DataArray(
        projection_data,
        dims=['channel', 'y', 'x'],
        coords={
            'channel': base_channels,
            'y': y_coords,
            'x': x_coords,
        },
        attrs={
            'description': 'Multi-channel equirectangular projections from point cloud data',
            'dataset_name': dataset_name,
            'canvas_size': canvas_size,
            'angular_resolution': angular_res,
        }
    )

    return df_filtered_ncolored, projection_xr


def _create_segmentation_masks(
    df: pd.DataFrame, 
    canvas_size: Tuple[int, int], 
    grouped: pd.core.groupby.DataFrameGroupBy
) -> Dict[str, np.ndarray]:
    """Create segmentation masks from class_id column.
    
    Args:
        df: DataFrame with class_id column
        canvas_size: Output image dimensions
        grouped: Grouped DataFrame for pixel coordinates
        
    Returns:
        Dictionary containing segmentation masks
    """
    seg_mask_raw = np.full(canvas_size, 255, dtype=np.int16)
    grouped_class_id = df.groupby(['y_pix', 'x_pix'], observed=False)['class_id'].agg(
        lambda x: x.value_counts().idxmax()
    )
    
    y_indices = grouped_class_id.index.get_level_values(0)
    x_indices = grouped_class_id.index.get_level_values(1)
    seg_mask_raw[y_indices, x_indices] = grouped_class_id.values
    
    # Handle special class values
    seg_mask_raw[seg_mask_raw == -1] = 18
    seg_mask_raw = seg_mask_raw.astype(np.uint8)
    
    # Create merged mask
    seg_mask_merged = seg_mask_raw.copy()
    seg_mask_merged[np.isin(seg_mask_merged, [18, 255])] = 17
    
    return {
        'seg_mask_raw': seg_mask_raw,
        'seg_mask_merged': seg_mask_merged
    }


def equirectangular_projection_normals(
    df_filtered_ncolored: pd.DataFrame,
    canvas_size: Tuple[int, int]
) -> np.ndarray:
    """Create RGB image from normal vector colors.
    
    Projects normal vector HSV colors onto a 2D equirectangular grid
    by averaging colors within each pixel.

    Args:
        df_filtered_ncolored: DataFrame with normal color columns (n_r, n_g, n_b)
        canvas_size: Output image dimensions (height, width)

    Returns:
        RGB image array of shape (height, width, 3) with normal-based colors
    """
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False)
    
    # Aggregate normal colors
    color_aggregation = {
        'n_r': grouped['n_r'].mean(),
        'n_g': grouped['n_g'].mean(),
        'n_b': grouped['n_b'].mean()
    }

    # Initialize RGB image arrays
    rgb_images = {
        channel: np.zeros(canvas_size, dtype=np.float32) 
        for channel in ['r', 'g', 'b']
    }

    # Get pixel indices
    pixel_indices = np.array(color_aggregation['n_r'].index.tolist())
    y_coords, x_coords = pixel_indices[:, 0], pixel_indices[:, 1]

    # Populate RGB channels
    rgb_images['r'][y_coords, x_coords] = color_aggregation['n_r'].values
    rgb_images['g'][y_coords, x_coords] = color_aggregation['n_g'].values
    rgb_images['b'][y_coords, x_coords] = color_aggregation['n_b'].values

    # Stack channels to create RGB image
    rgb_image = np.stack([rgb_images['r'], rgb_images['g'], rgb_images['b']], axis=-1)
    
    return rgb_image


def generate_2D_projection_images(
    output_dir: Path,
    canvas_size: Tuple[int, int], 
    angular_res: Tuple[int, int], 
    dataset_name: str = 'MANGROVE',
    color_map: Optional[Dict[str, tuple]] = None,
    saveflag: bool = False,
    visualize: bool = True,
    extra_maps: bool = True,
    show_single_band: bool = True,
    show_pseudo_rgb: bool = True,
    show_pca: bool = True,
    v_fov: Tuple[float, float] = (0, 90),
    h_fov: Tuple[float, float] = (0, 360)
) -> None:
    """Generate comprehensive 2D projection images from point cloud data.
    
    Creates various visualization outputs including single-band images,
    pseudo-RGB combinations, PCA components, and correlation matrices.

    Args:
        output_dir: Directory containing point cloud data and for outputs
        canvas_size: Output image dimensions (height, width)
        angular_res: Angular resolution (vertical, horizontal) in degrees
        dataset_name: Dataset type ('MANGROVE' or 'SEMANTIC3D')
        color_map: Color mapping for segmentation visualization
        saveflag: Whether to save generated images
        visualize: Whether to display images
        extra_maps: Whether to generate additional feature maps
        show_single_band: Whether to show individual band images
        show_pseudo_rgb: Whether to generate pseudo-RGB combinations
        show_pca: Whether to show PCA/MNF/ICA components
        v_fov: Vertical field of view range (min, max) in degrees
        h_fov: Horizontal field of view range (min, max) in degrees
        
    Raises:
        FileNotFoundError: If required input files are not found
    """
    # Setup paths
    input_file_stem = output_dir.parent.name
    pcd_dir = output_dir / 'pcd'
    img_out_dir = output_dir / 'img'
    
    create_dir_if_not_exists(img_out_dir, ask_user=False)
    
    # Find input file
    filename = next(pcd_dir.glob("*_ncr_0.02*"), None)
    if filename is None:
        raise FileNotFoundError(f"No file matching '*_ncr_0.02*' found in {pcd_dir}")

    # Generate projections
    key_str = f"{input_file_stem.split('_')[0]}_{input_file_stem.split('_')[-1]}"
    df_filtered, projection_xr = equirectangular_projection_multi(
        filename, canvas_size, angular_res, dataset_name
    )
    normals_rgb_image = equirectangular_projection_normals(df_filtered, canvas_size)

    # Generate visualizations using xarray directly
    # Save image cube and metadata
    image_cube, _ = save_image_cube_and_meta(
        projection_xr, normals_rgb_image, img_out_dir, key_str
    )

    # Generate correlation matrix
    generate_correlation_matrix(
        projection_xr, normals_rgb_image, img_out_dir, key_str
    )

    # Generate dataset-specific outputs
    if dataset_name == 'SEMANTIC3D':
        generate_semantic3d_outputs(
            projection_xr, df_filtered, img_out_dir, key_str, 
            color_map, saveflag, visualize, canvas_size, v_fov, h_fov
        )

    if extra_maps:
        generate_extra_visualizations(
            projection_xr, normals_rgb_image, image_cube, img_out_dir, 
            key_str, saveflag, visualize, show_single_band, show_pseudo_rgb, 
            show_pca, canvas_size, v_fov, h_fov
        )


def main() -> None:
    """Main function to run spherical projection processing."""
    params = CONFIG['spherical_projection']
    global_config = CONFIG['global']
    
    # Extract parameters
    v_fov = global_config['v_fov']
    h_fov = global_config['h_fov']
    canvas_size = (
        int((v_fov[1] - v_fov[0]) / global_config['v_ang_res_deg']),
        int((h_fov[1] - h_fov[0]) / global_config['h_ang_res_deg'])
    )
    angular_res = (global_config['v_ang_res_deg'], global_config['h_ang_res_deg'])
    
    # Process each output directory
    for output_dir in global_config['output_dir_ls']:
        generate_2D_projection_images(
            output_dir=output_dir,
            canvas_size=canvas_size, 
            angular_res=angular_res, 
            dataset_name=global_config['dataset'],
            color_map=global_config['color_map'],
            saveflag=params['save_extra_maps'],
            visualize=params['visualize'],
            extra_maps=params['extra_maps'],
            show_single_band=params['show_single_band'],
            show_pseudo_rgb=params['show_pseudo_rgb'],
            show_pca=params['show_pca'],
            v_fov=v_fov,
            h_fov=h_fov
        )


if __name__ == "__main__":
    main()