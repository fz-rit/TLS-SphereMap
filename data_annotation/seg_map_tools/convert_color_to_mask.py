import re
import argparse
import numpy as np
import json
from PIL import Image
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
current_file_dir = current_file.parent
parent_dir = current_file.parents[2]  # Go two levels up to root directory
sys.path.append(str(parent_dir))
sys.path.append(str(current_file_dir))

from tools.config_loader import get_config

# Load label maps from JSON file
def load_dataset_info(dataset_name):
    with open(current_file_dir / "label_maps.json", "r") as file:
        label_maps = json.load(file)["DATASETS"]

    if dataset_name not in label_maps:
        raise ValueError(f"Dataset '{dataset_name}' not found in label_maps.json.")

    # Load color-to-index mapping
    color_to_index = {tuple(map(int, k.split(","))): v for k, v in label_maps[dataset_name]["COLOR_TO_INDEX"].items()}

    # Load class names
    class_names = {int(k): v for k, v in label_maps[dataset_name]["CLASS_NAMES"].items()}

    return color_to_index, class_names

def convert_color_to_mask(seg_map_dir=None, dataset_name=None):
    """Convert a colorful segmentation map to a grayscale class index mask."""
    
    # pattern = re.compile(r"seg_map_.*_(\d{4})\.png")
    # # Check if the directory contains any file with the pattern "_mask", delete it if exists
    # mask_files = list(seg_map_dir.glob("*_mask*"))
    # for mask_file in mask_files:
    #     if mask_file.is_file():
    #         print(f"Deleting existing mask file: {mask_file}")
    #         mask_file.unlink()


    input_map = next((p for p in seg_map_dir.glob("seg_map*.png") 
                     if "_mask" not in p.stem), None)
    if not input_map:
        raise FileNotFoundError(f"No segmentation map found in {seg_map_dir} matching the pattern.")
    # input_map = [p for p in seg_map_dir.glob("seg_map*.png") if pattern.fullmatch(p.name)]
    # if not input_map:
    #     raise FileNotFoundError(f"No segmentation map found in {seg_map_dir} matching the pattern.")
    # elif len(input_map) > 1:
    #     raise ValueError(f"Multiple segmentation maps found in {seg_map_dir}. Please ensure only one matches the pattern.")
    # input_map = input_map[0]

    save_path = seg_map_dir / f"{input_map.stem}_mask.png"
    color_to_index, class_names = load_dataset_info(dataset_name)

    # Load the colorful PNG file
    img = Image.open(input_map).convert("RGB")
    img_np = np.array(img)  # Convert to NumPy array (H, W, 3)

    class_map = np.zeros(img_np.shape[:2], dtype=np.uint8)

    # Convert RGB colors to class indices
    for color, class_id in color_to_index.items():
        mask = np.all(np.abs(img_np - np.array(color)) <= 3, axis=-1)
        class_map[mask] = class_id

    # Convert to PIL Image and save the class index mask
    mask_img = Image.fromarray(class_map)
    mask_img.save(save_path)

    print(f"Converted segmentation map saved as {save_path}")

    # Display class names for verification
    print("\nClass Labels Used:")
    for idx, name in class_names.items():
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    current_config = get_config()
    if current_config is None:
        print("❌ Error: Configuration not loaded. Please run this script through run_3d_to_2d_pipeline.py")
        print("   Example: python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic_rc.json")
        exit(1)
    
    dataset_name = current_config["convert_color_to_mask"]["dataset"]
    output_dir_ls = current_config["global"]["output_dir_ls"]
    for directory in output_dir_ls:
        seg_map_dir = directory / "img"
        convert_color_to_mask(seg_map_dir=seg_map_dir, dataset_name=dataset_name)
