"""
This script processes point cloud data by attaching color and segmentation labels from corresponding segmentation maps.
It reads point cloud data and segmentation maps, attaches RGB values and segmentation labels to the points, and saves the
resulting data to a file.
Functions:
- attach_seg_map_colors_to_points(pts_df: pd.DataFrame, seg_map: np.ndarray) -> pd.DataFrame:
- convert_gray_to_label_map(gray_image: np.ndarray) -> np.ndarray:
- attach_seg_map_label_to_points(pts_df: pd.DataFrame, seg_map_gray: np.ndarray) -> pd.DataFrame:
- attach_segmentation_color_label_to_points(pt_cloud_path: Path, seg_map_rgb_path: Path, seg_map_mono_path: Path, output_dir: Path) -> pd.DataFrame:
    Attach the color and label of the pixels in the segmentation map back to the points in the point cloud DataFrame and save the result.
Main Execution:
---------------
- Reads configuration from a JSON file.
- Processes the point cloud data by attaching color and segmentation labels.
- Saves the processed data to a specified output directory.

To use this script, specify the input paths in a JSON file and run the script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
import json
from preprocess_point_cloud import preprocess_point_cloud, read_point_cloud
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import cv2
from convert_grayscale_to_label_map import convert_grayscale_to_label_map

def attach_seg_map_colors_to_points(pts_df: pd.DataFrame, 
                                         seg_map: np.ndarray) -> pd.DataFrame:
    """
    Attach the color of the pixels in the segmentation map back to the points in the point cloud DataFrame.

    Parameters:
    -----------
    pts_df : pd.DataFrame
        The filtered point cloud DataFrame with pixel coordinates.
    seg_map_path : Path
        The path to the segmentation map RGB image file.
    output_file_path : Path
        The path to save the colorized point cloud data.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with additional columns for RGB values.
    """
    # Attach the color of the pixels in the segmentation map back to the points in the point cloud DataFrame

    # Create new columns in the DataFrame for RGB values and initialize them
    pts_df.loc[:, 'red'] = np.nan
    pts_df.loc[:, 'green'] = np.nan
    pts_df.loc[:, 'blue'] = np.nan

    # Extract unique pixel coordinates and map their RGB values
    pxpy_indices = pts_df[['x_pix', 'y_pix']].drop_duplicates().values
    rgb_values = seg_map[pxpy_indices[:, 1], pxpy_indices[:, 0]]  # Access using (row, column)
    coord_to_rgb = dict(zip(map(tuple, pxpy_indices), rgb_values))

    # Apply RGB values to the DataFrame using a vectorized operation
    coords = pts_df[['x_pix', 'y_pix']].apply(tuple, axis=1)
    rgb_data = coords.map(coord_to_rgb)

    # Convert the mapped RGB data to a DataFrame
    rgb_df = pd.DataFrame(rgb_data.tolist(), columns=['red', 'green', 'blue'])
    
    # Assign RGB values to the corresponding columns
    pts_df[['red', 'green', 'blue']] = rgb_df

    
    return pts_df




# def convert_gray_to_label_map(gray_image: np.ndarray) -> np.ndarray:
#     """
#     Convert a grayscale image to a segmentation label map with labels ranging from 0 to 4.
#     Outlier pixels are assigned labels based on the spatially nearest labeled pixel to promote
#     connectivity of regions. This implementation uses OpenCV for efficiency and conciseness.

#     Parameters
#     ----------
#     gray_image : np.ndarray
#         Input grayscale image as a NumPy array.

#     Returns
#     -------
#     label_map : np.ndarray
#         Segmentation label map with integer labels from 0 to 4.
#     """
#     # Define the label-to-grayscale-values mapping with approximate values
#     label_to_gray_values = {
#         0: [0, 1], # Void
#         1: [99, 100, 101], # Miscellaneous
#         2: [121, 122, 123], # Leaves
#         3: [140, 141, 142], # Bark
#         4: [233, 234, 235], # Soil
#     }

#     # Build a grayscale value to label mapping
#     gray_value_to_label = {}
#     for label, gray_values in label_to_gray_values.items():
#         for gray_value in gray_values:
#             gray_value_to_label[gray_value] = label

#     # Initialize the label map with -1, using a signed integer type
#     label_map = np.full_like(gray_image, fill_value=-1, dtype=np.int32)

#     # Apply the mapping to the image
#     for gray_value, label in gray_value_to_label.items():
#         label_map[gray_image == gray_value] = label

#     # Create a mask of valid labels
#     valid_mask = label_map >= 0

#     # Check if there are any outliers to process
#     if np.any(~valid_mask):
#         # Invert the valid mask for OpenCV distance transform (foreground pixels are non-zero)
#         inverted_mask = (~valid_mask).astype(np.uint8)

#         # Perform distance transform and get labels of nearest valid pixels
#         distance, labels = cv2.distanceTransformWithLabels(
#             inverted_mask,
#             distanceType=cv2.DIST_L2,
#             maskSize=5,
#             labelType=cv2.DIST_LABEL_PIXEL
#         )

#         # Adjust labels to get indices of nearest valid pixels
#         nearest_labels = labels - 1  # OpenCV labels start from 1

#         # Get coordinates of all valid pixels
#         valid_coords = np.column_stack(np.nonzero(valid_mask))

#         # Map labels to outlier pixels based on nearest valid pixel
#         outlier_coords = np.column_stack(np.nonzero(~valid_mask))
#         nearest_valid_indices = nearest_labels[~valid_mask].astype(np.int32)

#         # Ensure indices are within valid range
#         nearest_valid_indices = np.clip(nearest_valid_indices, 0, len(valid_coords) - 1)

#         # Assign labels from nearest valid pixels to outlier pixels
#         nearest_valid_coords = valid_coords[nearest_valid_indices]
#         label_map[~valid_mask] = label_map[tuple(nearest_valid_coords.T)]

#     return label_map.astype(np.uint8)  # Convert to uint8 for consistent data type



def attach_seg_map_label_to_points(pts_df: pd.DataFrame, 
                                   seg_map_gray: np.ndarray) -> pd.DataFrame:
    """
    Attach the label of the pixels in the segmentation map back to the points in the point cloud DataFrame.

    """
    # Attach the label of the pixels in the segmentation map back to the points in the point cloud DataFrame 
    # Convert the grayscale image to a segmentation map with unique labels
    seg_map_label = convert_grayscale_to_label_map(seg_map_gray)

    # Create new columns in the DataFrame for RGB values and initialize them
    pts_df.loc[:, 'seg_label'] = np.nan

    # Extract unique pixel coordinates and map their RGB values
    pxpy_indices = pts_df[['x_pix', 'y_pix']].drop_duplicates().values
    labels = seg_map_label[pxpy_indices[:, 1], pxpy_indices[:, 0]]  # Access using (row, column)
    coord_to_rgb = dict(zip(map(tuple, pxpy_indices), labels))

    # Apply RGB values to the DataFrame using a vectorized operation
    coords = pts_df[['x_pix', 'y_pix']].apply(tuple, axis=1)
    labels_column = coords.map(coord_to_rgb)

    # Convert the mapped RGB data to a DataFrame
    label_df = pd.DataFrame(labels_column.tolist(), columns=['seg_label'])
    
    # Assign RGB values to the corresponding columns
    pts_df['seg_label'] = label_df


    return pts_df


def attach_segmentation_color_label_to_points(pt_cloud_path: Path, 
                                              seg_map_rgb_path: Path, 
                                              seg_map_mono_path: Path, 
                                              output_dir: Path) -> pd.DataFrame:
    """
    Attach the color and label of the pixels in the segmentation map back to the points in the point cloud DataFrame.

    Parameters:
    -----------
    pt_cloud_path : Path
        The path to the point cloud data file in either .las or .txt format.
    seg_map_rgb_path : Path
        The path to the segmentation map RGB image file.
    seg_map_mono_path : Path
        The path to the segmentation map grayscale image file.
    output_dir : Path
        The directory to save the colorized point cloud data.
    
    Returns:
    --------
    pd.DataFrame
        The DataFrame with additional columns for RGB values and segmentation labels.
    """
    # Read the point cloud data
    if pt_cloud_path.suffix == '.las':
        pts_df = preprocess_point_cloud(pt_cloud_path, clean_pc=False, upside_down=False)
    elif pt_cloud_path.suffix == '.txt':
        pts_df = read_point_cloud(pt_cloud_path)
    else:
        raise ValueError(f"Unsupported file extension: {pt_cloud_path.suffix}")
    pts_df = pts_df.reset_index(drop=True)
    seg_map_rgb = io.imread(seg_map_rgb_path)  # (height, width, rgb-channels)
    seg_map_gray = io.imread(seg_map_mono_path)  # (height, width)

    # Attach the color and label of the segmentation map back to the points in the point cloud DataFrame
    pts_df_colored = attach_seg_map_colors_to_points(pts_df, seg_map_rgb)
    pts_df_colored_labeled = attach_seg_map_label_to_points(pts_df_colored, seg_map_gray)

    # Save the colorized point cloud to a text file
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f'Created output directory: {output_dir}!!!')
    output_file_path = output_dir / f'{pt_cloud_path.stem}_wt_segmap.txt'
    pts_df_colored_labeled.to_csv(output_file_path, sep=',', index=False)
    print(f'Colorized point cloud data saved to {output_file_path}!')

    return pts_df_colored_labeled


if __name__ == '__main__':

    json_path = Path('./input_params/attach_segmap_to_points_amiri.json')
    with open(json_path, 'r') as file:
        config = json.load(file)

    root_dir = Path(config["root_dir"])
    pt_cloud_path = root_dir / config["pt_cloud_filename"]
    seg_map_rgb_path = root_dir / config["seg_map_rgb_filename"]
    seg_map_mono_path = root_dir / config["seg_map_mono_filename"]
    output_dir = Path(config["output_dir"])

    attach_segmentation_color_label_to_points(pt_cloud_path, seg_map_rgb_path, seg_map_mono_path, output_dir)