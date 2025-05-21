import json
import yaml
from pathlib import Path
from pprint import pprint

# --- Paths ---
json_dir = Path("/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/inlut3d")
json_path = json_dir / "train_val_test_split_inlut3d.json"

output_yaml_path = json_dir / "concrete_paths_inlut3d.yaml"
root_dir = Path("/media/fzhcis/Seagate Expansion Drive/point_cloud_data/outputs/inlut3d")
# --- Load and sort ID lists ---
with open(json_path, 'r') as f:
    id_dict = json.load(f)
for split in ["train", "val", "test"]:
    id_dict[split].sort()
# pprint(id_dict)

# --- Temporary match storage ---
temp_output = {
    "img": {"train": {}, "val": {}, "test": {}},
    "img_metadata": {"train": {}, "val": {}, "test": {}},
    "mask": {"train": {}, "val": {}, "test": {}},
    # "pcd": {"train": {}, "val": {}, "test": {}}
}

# --- Match .npy files in 'img' folder ---
for npy_file in root_dir.rglob("*.npy"):
    parent_dir_name = npy_file.parent.name
    match_str = npy_file.parent.parent.name
    if parent_dir_name == "img":
        for split in ["train", "val", "test"]:
            if match_str in id_dict[split]:
                temp_output["img"][split][match_str] = str(npy_file.resolve())
                break

# --- Match .yaml files in 'img' folder ---
for yaml_file in root_dir.rglob("*.yaml"):
    parent_dir_name = yaml_file.parent.name
    match_str = yaml_file.parent.parent.name
    if parent_dir_name == "img":
        for split in ["train", "val", "test"]:
            if match_str in id_dict[split]:
                temp_output["img_metadata"][split][match_str] = str(yaml_file.resolve())
                break

# --- Match .png files in 'img' folder ---
for png_file in root_dir.rglob("seg_map_merged*_mask.png"):
    parent_dir_name = png_file.parent.name
    match_str = png_file.parent.parent.name
    if parent_dir_name == "img":
        for split in ["train", "val", "test"]:
            if match_str in id_dict[split]:
                temp_output["mask"][split][match_str] = str(png_file.resolve())
                break

# # --- Match .csv files in 'pcd' folder ---
# for csv_file in root_dir.rglob("*_color.csv"):
#     if csv_file.parent.name == "pcd":
#         match_str = csv_file.name.split("color")[0][-5:]
#         # print(f"Matching {match_str} in {csv_file}")
#         for split in ["train", "val", "test"]:
#             if match_str in id_dict[split]:
#                 temp_output["pcd"][split][match_str] = str(csv_file.resolve())
#                 break

# --- Construct final sorted output ---
output = {
    "img": {
        split: [temp_output["img"][split][id_] for id_ in id_dict[split] if id_ in temp_output["img"][split]]
        for split in ["train", "val", "test"]
    },
    "img_metadata": {
        split: [temp_output["img_metadata"][split][id_] for id_ in id_dict[split] if id_ in temp_output["img_metadata"][split]]
        for split in ["train", "val", "test"]
    },
    "mask": {
        split: [temp_output["mask"][split][id_] for id_ in id_dict[split] if id_ in temp_output["mask"][split]]
        for split in ["train", "val", "test"]
    },
    # "pcd": {
    #     split: [temp_output["pcd"][split][id_] for id_ in id_dict[split] if id_ in temp_output["pcd"][split]]
    #     for split in ["train", "val", "test"]
    # }
}

# --- Save to YAML ---
with open(output_yaml_path, 'w') as f:
    yaml.dump(output, f, default_flow_style=False)

print(f"Saved structured and sorted paths to {output_yaml_path}")
for mode in output.keys():
    for split in ["train", "val", "test"]:
        print(f"  {mode}/{split}: {len(output[mode][split])} files")


# --- Load and print the YAML file ---
def validate_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)


    # Example: access train image paths
    train_img_paths = data["img"]["train"]
    print(f"\nNumber of train .npy images: {len(train_img_paths)}")
    print("First few .npy paths:")
    pprint(train_img_paths[:3])


validate_yaml(output_yaml_path)