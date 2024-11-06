from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from preprocess_point_cloud import preprocess_point_cloud


def attach_segmentation_colors_to_points(pts_df: pd.DataFrame, seg_map_path: Path, output_file_path: Path) -> pd.DataFrame:
    """
    Attach the color of the pixels in the segmentation map back to the points in the point cloud DataFrame.

    Parameters:
    -----------
    pts_df : pd.DataFrame
        The filtered point cloud DataFrame with pixel coordinates.
    seg_map_path : Path
        The path to the segmentation map image file.
    output_file_path : Path
        The path to save the colorized point cloud data.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with additional columns for RGB values.
    """
    # Step 8: Attach the color of the pixels in the segmentation map back to the points
    ## read the segmented image
    seg_map = io.imread(seg_map_path)  # (height, width, rgb-channels)

    # Create new columns in the DataFrame for RGB values and initialize them
    pts_df.loc[:, 'red'] = np.nan
    pts_df.loc[:, 'green'] = np.nan
    pts_df.loc[:, 'blue'] = np.nan

    # Extract unique pixel coordinates and map their RGB values
    pxpy_indices = pts_df[['y_pix', 'x_pix']].drop_duplicates().values
    rgb_values = seg_map[pxpy_indices[:, 0], pxpy_indices[:, 1]]
    coord_to_rgb = dict(zip(map(tuple, pxpy_indices), rgb_values))

    # Apply RGB values to the DataFrame using a vectorized operation
    coords = pts_df[['y_pix', 'x_pix']].apply(tuple, axis=1)
    rgb_data = coords.map(coord_to_rgb)

    # Assign RGB values to the corresponding columns
    pts_df[['red', 'green', 'blue']] = pd.DataFrame(rgb_data.tolist(), index=pts_df.index)

    # Save the colorized point cloud to a text file
    pts_df.to_csv(output_file_path, sep='\t', index=False)
    print(f'Colorized point cloud data saved to {output_file_path}!')
    
    return pts_df

if __name__ == '__main__':
    pt_cloud_path = Path(r'C:\Users\fzhcis\Documents\projects\from_RobC\for_Fei\data\palau_2024\ALRSET1\UMBCBL009_2024-03-28-02-47-26_ALRSET12_060180_000200.800_1830507489.txt')
    # seg_map_path = Path(r'C:\Users\fzhcis\Documents\mylab\tls_demo\outputs\UMBCBL009_2024-03-28-02-47-26_ALRSET12_060180_000200.800_1830507489\Intensity-map-root-segmentation-bitmap_0005_Layer 6 copy.tif')
    # seg_map_path = Path(r'C:\Users\fzhcis\Documents\mylab\tls_demo\outputs\UMBCBL009_2024-03-28-02-47-26_ALRSET12_060180_000200.800_1830507489\Pseudo-RGB_Intensity-Range-Density_0004_segmentation map.tif')
    seg_map_path = Path(r'C:\Users\fzhcis\Documents\mylab\tls_demo\tls_point_segmentation\outputs\Pseudo-RGB_Intensity-Range-Z.tif')
    output_dir = Path('./outputs')
    df_filtered = preprocess_point_cloud(pt_cloud_path)

    output_file_path = output_dir / f'{pt_cloud_path.stem}_2d_segmap_labeled.txt'
    pts_df_colored = attach_segmentation_colors_to_points(df_filtered, seg_map_path, output_file_path)