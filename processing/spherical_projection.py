import pandas as pd
import numpy as np
from tools.hdr_adjustments import contrast_enhancement
from tools.preprocess_point_cloud import map_angle_to_pixel
from typing import Union, List, Dict, Any, Tuple, Optional
from pathlib import Path
import xarray as xr
from tools.norm_to_hsv import attach_normal_color_to_df
from tools.config_loader import get_config
from tools.pcd_utils import create_dir_if_not_exists

from tools.spherical_projection_helper import (
    save_image_cube_and_meta,
    generate_correlation_matrix,
    generate_semantic3d_outputs,
    generate_forestsemantic_outputs,
    generate_extra_visualizations,
    select_by_min_range,
    select_min_range_and_count,
)

import pandas as pd
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

    # Normalize dataset name for flexible comparison
    dataset_upper = dataset_name.upper()
    
    # Define aggregation strategy based on dataset
    dataset_upper = dataset_name.upper()
    if 'SEMANTIC3D' in dataset_upper:
        float_cols = ['r', 'g', 'b']
        for col in float_cols:
            df_filtered_ncolored[col] = (df_filtered_ncolored[col] / 255.0).astype(np.float32)
        print("True RGB channels normalized to [0, 1] for SEMANTIC3D.")
    elif not any(x in dataset_upper for x in ['MANGROVE', 'FORESTSEMANTIC']):
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: 'SEMANTIC3D', 'MANGROVE', 'ForestSemantic'."
        )

    grouped_with_density = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).apply(
        select_min_range_and_count, include_groups=False
    )
    
    # Extract density and main data
    pts_per_pixel = grouped_with_density['point_count']
    grouped = grouped_with_density.drop(columns=['point_count'])

    # Define channel names and initialize data
    base_channels = [
        'intensity_raw', 'z_raw', 'range_raw',
        'intensity_adjusted', 'z_adjusted', 'range_adjusted',
        'curvature', 'anisotropy', 'planarity', 'density'
    ]
    
    # Add dataset-specific channels
    if 'SEMANTIC3D' in dataset_upper:
        base_channels.extend(['true_r', 'true_g', 'true_b'])
        if 'Classification' in df_filtered_ncolored.columns:
            base_channels.extend(['seg_mask'])
        else:
            print("Warning: 'Classification' column not found in SEMANTIC3D data. Skipping segmentation masks.")

    if 'FORESTSEMANTIC' in dataset_upper:
        if 'Classification' in df_filtered_ncolored.columns:
            base_channels.extend(['seg_mask'])

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
    if 'SEMANTIC3D' in dataset_upper:
        beautiful_colors = {"Sky ash": (160, 190, 220), "Periwinkle Mist": (180, 170, 255)}
        red, green, blue = (np.full(canvas_size, val / 255.0, dtype=np.float32) 
                    for val in beautiful_colors["Periwinkle Mist"])
        
        red[y_indices, x_indices] = grouped['r'].values
        green[y_indices, x_indices] = grouped['g'].values
        blue[y_indices, x_indices] = grouped['b'].values
        
        channel_mapping.update({
            'true_r': red,
            'true_g': green,
            'true_b': blue,
        })

        # Add segmentation masks for training data
        if 'Classification' in df_filtered_ncolored.columns:
            seg_mask = _create_segmentation_masks_semantic3d(
                df_filtered_ncolored, canvas_size
            )
            channel_mapping['seg_mask'] = seg_mask.astype(np.float32)
        else:
            print("Warning: 'Classification' column not found in SEMANTIC3D data. Skipping segmentation masks.")
            
    elif 'FORESTSEMANTIC' in dataset_upper:
        # Create segmentation mask if available
        if 'Classification' not in df_filtered_ncolored.columns:
            raise ValueError("❗ 'Classification' column not found in ForestSemantic data. ")
        seg_mask = _create_segmentation_masks_forestsemantic(
            df_filtered_ncolored, canvas_size
        )
        channel_mapping['seg_mask'] = seg_mask.astype(np.float32)

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


def _create_segmentation_masks_semantic3d(
    df: pd.DataFrame, 
    canvas_size: Tuple[int, int], 
) -> Dict[str, np.ndarray]:
    """Create segmentation masks from class_id column using nearest point aggregation.
    
    Uses the same nearest-point (minimum range) strategy as other features
    to ensure consistency across all projected attributes.
    
    Args:
        df: DataFrame with class_id column and pixel coordinates
        canvas_size: Output image dimensions
        dataset_name: Dataset name to determine class column
        
    Returns:
        Dictionary containing segmentation masks
    """
    

    # Use nearest point aggregation (same as other features)
    grouped_class_id = df.groupby(['y_pix', 'x_pix'], observed=False).apply(
        select_by_min_range, include_groups=False
    )['Classification']
    void_class_id = 9 # Hardcoded for Semantic3D
    seg_mask = np.full(canvas_size, void_class_id, dtype=np.uint8)
    y_indices = grouped_class_id.index.get_level_values(0)
    x_indices = grouped_class_id.index.get_level_values(1)
    seg_mask[y_indices, x_indices] = grouped_class_id.values
    
    return seg_mask

def _create_segmentation_masks_forestsemantic(
    df: pd.DataFrame, 
    canvas_size: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """Create segmentation masks for ForestSemantic dataset."""
    seg_mask = np.zeros(canvas_size, dtype=np.uint8)
    
    grouped_class_id = df.groupby(['y_pix', 'x_pix'], observed=False).apply(
        select_by_min_range, include_groups=False
    )["Classification"]
    
    y_indices = grouped_class_id.index.get_level_values(0)
    x_indices = grouped_class_id.index.get_level_values(1)
    seg_mask[y_indices, x_indices] = grouped_class_id.values
    
    return seg_mask


def equirectangular_projection_normals(
    df_filtered_ncolored: pd.DataFrame,
    canvas_size: Tuple[int, int]
) -> np.ndarray:
    """Create RGB image from normal vector colors.
    
    Projects normal vector HSV colors onto a 2D equirectangular grid
    by selecting normal colors from the point with minimum range within each pixel.

    Args:
        df_filtered_ncolored: DataFrame with normal color columns (n_r, n_g, n_b)
        canvas_size: Output image dimensions (height, width)

    Returns:
        RGB image array of shape (height, width, 3) with normal-based colors
    """
    # Use the same nearest point aggregation strategy as other features
    grouped = df_filtered_ncolored.groupby(['y_pix', 'x_pix'], observed=False).apply(
        select_by_min_range, include_groups=False
    )
    
    # Extract normal colors from the selected points
    normal_colors = grouped[['n_r', 'n_g', 'n_b']]

    # Initialize RGB image arrays
    rgb_images = {
        channel: np.zeros(canvas_size, dtype=np.float32) 
        for channel in ['r', 'g', 'b']
    }

    # Get pixel indices from the grouped normal colors
    pixel_indices = np.array(normal_colors.index.tolist())
    y_coords, x_coords = pixel_indices[:, 0], pixel_indices[:, 1]

    # Populate RGB channels using the selected normal colors
    rgb_images['r'][y_coords, x_coords] = normal_colors['n_r'].values
    rgb_images['g'][y_coords, x_coords] = normal_colors['n_g'].values
    rgb_images['b'][y_coords, x_coords] = normal_colors['n_b'].values

    # Stack channels to create RGB image
    rgb_image = np.stack([rgb_images['r'], rgb_images['g'], rgb_images['b']], axis=-1)
    
    return rgb_image


def generate_2D_projection_images(
    output_dir: Path,
    canvas_size: Tuple[int, int], 
    angular_res: Tuple[int, int], 
    dataset_name: str = 'MANGROVE',
    out_key_str: str = 'geom_feat',
    input_file_stem: str = 'pcd',
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
    # input_file_stem = output_dir.parent.name
    pcd_dir = output_dir / 'pcd'
    img_out_dir = output_dir / 'img'
    
    create_dir_if_not_exists(img_out_dir, ask_user=False)
    
    # Find input file
    filename = next(pcd_dir.glob(f"*{out_key_str}*"), None)
    if filename is None:
        raise FileNotFoundError(f"No file matching '*{out_key_str}*' found in {pcd_dir}")

    # Generate projections
    # key_str = f"{input_file_stem.split('_')[0]}_{input_file_stem.split('_')[-1]}"
    key_str = input_file_stem
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
    if 'SEMANTIC3D' in dataset_name.upper():
        generate_semantic3d_outputs(
            projection_xr, df_filtered, img_out_dir, key_str, 
            color_map, saveflag, visualize, canvas_size, v_fov, h_fov
        )
    elif 'FORESTSEMANTIC' in dataset_name.upper():
        generate_forestsemantic_outputs(
            projection_xr, df_filtered, img_out_dir, key_str, 
            color_map, saveflag, visualize, 
            # canvas_size, v_fov, h_fov
        )

    if extra_maps:
        generate_extra_visualizations(
            projection_xr, normals_rgb_image, image_cube, img_out_dir, 
            key_str, saveflag, visualize, show_single_band, show_pseudo_rgb, 
            show_pca, canvas_size, v_fov, h_fov
        )


def main() -> None:
    """Main function to run spherical projection processing."""
    # Check if CONFIG is loaded, try auto-load if not
    current_config = get_config()
    
    if current_config is None:
        print("❌ Error: Configuration not loaded. Please run this script through run_3d_to_2d_pipeline.py")
        print("   Example: python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic_rc.json")
        return
        
    params = current_config['spherical_projection']
    global_config = current_config['global']
    input_path_ls = global_config['input_path_ls']
    output_dir_ls = global_config['output_dir_ls']
    v_fov = global_config['v_fov']
    h_fov = global_config['h_fov']
    canvas_size = global_config['canvas_size']
    angular_res = (global_config['v_ang_res_deg'], global_config['h_ang_res_deg'])
    
    # for output_dir in global_config['output_dir_ls']:
    for input_path, output_dir in zip(input_path_ls, output_dir_ls):
        generate_2D_projection_images(
            output_dir=output_dir,
            canvas_size=canvas_size, 
            angular_res=angular_res, 
            dataset_name=global_config['dataset'],
            out_key_str=current_config['calc_geom_feature']['out_signature_str'],
            input_file_stem=input_path.stem,
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