"""
taking inputs: 
    pcd .csv file - calculate px & py from zenith and azimuth; 
    original label; 
    refined label; 
    original 2D mask;
    color map
output: refined 2D mask.
track: relabeled points; relabeld pixels; difference map.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from tools.preprocess_point_cloud import map_angle_to_pixel
from PIL import Image
import json
from tools.config_loader import CONFIG
import matplotlib.pyplot as plt




def map_diff_labels_to_pixels(point_cloud_file, original_label_file, refined_label_file, angular_res, canvas_size):
    
    pc_df = pd.read_csv(point_cloud_file, sep=',')
    azimuth, elevation = pc_df['azimuth'], pc_df['elevation']

    x_pix, y_pix = map_angle_to_pixel(azimuth, elevation, 
                                      angular_res=angular_res, 
                                      canvas_size=canvas_size)
    pc_df['x_pix'] = x_pix.astype(int)
    pc_df['y_pix'] = y_pix.astype(int)

    original_laebl = np.loadtxt(original_label_file, dtype=int)
    refined_label = np.loadtxt(refined_label_file, dtype=int)

    pc_df['original_label'] = original_laebl
    pc_df['refined_label'] = refined_label

    out_df = pc_df[['x_pix', 'y_pix', 'original_label', 'refined_label']].copy()
    diff_df = out_df[out_df['original_label'] != out_df['refined_label']]

    print(f"Number of relabeld points: {len(diff_df)}")

    return out_df, diff_df

# def load_color_map(label_maps_path, dataset_name = "MANGROVE_ROOTS"):
#     with open(label_maps_path, "r") as file:
#         label_maps = json.load(file)["DATASETS"]

#     if dataset_name not in label_maps:
#         raise ValueError(f"Dataset '{dataset_name}' not found in label_maps.json.")

#     # Convert string keys to tuple format for colors
#     color_to_index = {tuple(map(int, k.split(","))): v for k, v in label_maps[dataset_name]["COLOR_TO_INDEX"].items()}
#     class_names = {int(k): v for k, v in label_maps[dataset_name]["CLASS_NAMES"].items()}
#     index_to_color = {v: k for k, v in color_to_index.items()}

#     return color_to_index, class_names, index_to_color



def load_segmentation_map(segmap_file):

    segmap = np.array(Image.open(segmap_file))  # Shape: (H, W)

    return segmap

def refine_segmentation_map(segmap, points_df):
    """
    Refine the segmentation map based on relabeled points.
    """
    refined_map = np.zeros_like(segmap)

    grouped_refined_label = points_df.groupby(['y_pix', 'x_pix'], observed=False)['refined_label'].agg(
        lambda x: x.value_counts().idxmax()
    )
    
    
    y_indices = grouped_refined_label.index.get_level_values(0)
    x_indices = grouped_refined_label.index.get_level_values(1)
    refined_map[y_indices, x_indices] = grouped_refined_label.values

    original_map_reconstructed = np.zeros_like(segmap)
    # grouped_original_label = points_df.groupby(['y_pix', 'x_pix'], observed=False)['original_label'].agg(
    #     lambda x: x.value_counts().idxmax()
    # )
    # original_y_indices = grouped_original_label.index.get_level_values(0)
    # original_x_indices = grouped_original_label.index.get_level_values(1)
    # original_map_reconstructed[original_y_indices, original_x_indices] = grouped_original_label.values

    return refined_map, original_map_reconstructed

def get_difference_map(original_map, refined_map, extra_str=""):
    """
    Create a difference map showing changes between original and refined segmentation maps.
    """
    diff_map = np.zeros_like(original_map)
    diff_map[original_map != refined_map] = 255  # Mark differences in white

    print(f"Number of relabeld pixels {extra_str}: {np.sum(diff_map > 0)}")
    return diff_map


def display_maps(original_map, refined_map, diff_map, save_path=None, compare_map_name="Refined"):
    """
    Display the original, refined, and difference maps.
    """
    plt.figure(figsize=(6, 8))
    
    plt.subplot(3, 1, 1)
    plt.title("Original Segmentation Map")
    plt.imshow(original_map, cmap='jet')
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.title(f"{compare_map_name} Segmentation Map")
    plt.imshow(refined_map, cmap='jet')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.title("Difference Map")
    plt.imshow(diff_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Maps saved to {save_path}")

def process_segmentation_map(mask_file, points_df):
    """
    Process the segmentation maps and visualize the results.
    """
    original_map = load_segmentation_map(mask_file)
    
    refined_map, _ = refine_segmentation_map(original_map, points_df)
    
    diff_map = get_difference_map(original_map, refined_map, extra_str="Org vs Refined")
    display_maps_save_path = mask_file.parent / f"{str(mask_file.stem.split('_mask')[0])}_segmk_refine_comp3in1.png"
    display_maps(original_map, refined_map, diff_map, save_path=display_maps_save_path, compare_map_name="Refined")
    

    # org_diff_map = get_difference_map(original_map, original_map_reconstructed, extra_str="Org vs Org-Reconstructed")
    # display_maps(original_map, original_map_reconstructed, org_diff_map, compare_map_name="Original Reconstructed")

    return refined_map, diff_map

def save_refined_map(mask_file, refined_map, diff_map):
    """
    Save the refined segmentation map to a file.
    """

    refine_map_file = mask_file.parent / f"{str(mask_file.stem.split('_mask')[0])}_segmk_refined.png"
    diff_map_file = mask_file.parent / f"{str(mask_file.stem.split('_mask')[0])}_segmk_diff.png"

    Image.fromarray(refined_map.astype(np.uint8)).save(refine_map_file)
    Image.fromarray(diff_map.astype(np.uint8)).save(diff_map_file)

    print(f"Refined map saved to {refine_map_file.name}")
    print(f"Difference map saved to {diff_map_file.name}")


global_params = CONFIG['global']
angular_res = (global_params['v_ang_res_deg'], global_params['h_ang_res_deg'])
canvas_size = global_params['canvas_size']
root_dir = Path("/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/palau_2024")
original_label_paths = list(root_dir.glob('**/*seg.label'))
refined_label_paths = list(root_dir.glob('**/*_refined.label'))
csv_paths = list(root_dir.glob('**/*_color.csv'))
mask_paths = list(root_dir.glob('**/img/seg_map_*_*_mask.png'))
original_label_paths.sort()
refined_label_paths.sort()
csv_paths.sort()
mask_paths.sort()
print(f"Number of original label files: {len(original_label_paths)} \n \
Number of refined label files: {len(refined_label_paths)} \n \
Number of CSV files: {len(csv_paths)} \n \
Number of mask files: {len(mask_paths)}")

for i in range(39):
    csv_path = csv_paths[i]
    original_label_path = original_label_paths[i]
    refined_label_path = refined_label_paths[i]
    mask_path = mask_paths[i]
    print(f"--------Processing {i:02d} {csv_path.name} with mask {mask_path.name}---------")
    label_df, label_diff_df = map_diff_labels_to_pixels(csv_path, 
                          original_label_path, 
                          refined_label_path, 
                          angular_res, canvas_size)
    refined_map, diff_map = process_segmentation_map(mask_path, label_df)
    save_refined_map(mask_path, refined_map, diff_map)
# plt.show()
