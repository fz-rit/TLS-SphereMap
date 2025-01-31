import argparse
import numpy as np
import json
from PIL import Image
import os

# Load label maps from JSON file
def load_dataset_info(dataset_name):
    with open("label_maps.json", "r") as file:
        label_maps = json.load(file)["DATASETS"]

    if dataset_name not in label_maps:
        raise ValueError(f"Dataset '{dataset_name}' not found in label_maps.json.")

    # Load color-to-index mapping
    color_to_index = {tuple(map(int, k.split(","))): v for k, v in label_maps[dataset_name]["COLOR_TO_INDEX"].items()}

    # Load class names
    class_names = {int(k): v for k, v in label_maps[dataset_name]["CLASS_NAMES"].items()}

    return color_to_index, class_names

def convert_color_to_mask(image_path, save_path, dataset_name):
    """Convert a colorful segmentation map to a grayscale class index mask."""
    color_to_index, class_names = load_dataset_info(dataset_name)

    # Load the colorful PNG file
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # Convert to NumPy array (H, W, 3)

    # Initialize the class index mask
    class_map = np.zeros(img_np.shape[:2], dtype=np.uint8)

    # Convert RGB colors to class indices
    for color, class_id in color_to_index.items():
        mask = np.all(img_np == np.array(color), axis=-1)
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
    # Load available datasets from JSON
    with open("label_maps.json", "r") as file:
        available_datasets = json.load(file)["DATASETS"].keys()

    parser = argparse.ArgumentParser(description="Convert a colorful segmentation PNG to a grayscale class index mask.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input colorful segmentation PNG.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output grayscale class index mask PNG.")
    parser.add_argument("-d", "--dataset", required=True, choices=available_datasets, help="Dataset name to select the appropriate colormap and class names.")

    args = parser.parse_args()
    convert_color_to_mask(args.input, args.output, args.dataset)
