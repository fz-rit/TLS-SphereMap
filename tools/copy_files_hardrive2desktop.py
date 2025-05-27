import shutil
import os
import yaml
from pprint import pprint
from pathlib import Path
import tqdm
        


def copy_files(source_file, destination_parent_dir, parent_dir_name=None):

    destination_directory = destination_parent_dir / parent_dir_name
    new_filename = f"{source_file.parent.parent.name}_{'_'.join(source_file.stem.split('_')[1:])}{source_file.suffix}"

    destination_file_path = os.path.join(destination_directory, new_filename)
    os.makedirs(destination_directory, exist_ok=True)

    try:
        shutil.copy2(source_file, destination_file_path)
        print(f"File '{source_file}' copied to '{destination_file_path}' and renamed successfully.")
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied. Check file permissions for '{source_file}' or '{destination_directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


yaml_path = "/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/inlut3d/concrete_paths_inlut3d.yaml"
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

destination_parent_dir = Path("/home/fzhcis/mylab/data/inlut3d_img_cube_2d_mask")
splits = ["train", "val", "test"]
parent_dirs = ["img", "img_metadata", "mask"]
for parent_dir in tqdm.tqdm(parent_dirs, desc="Parent Directories"):
    for split in tqdm.tqdm(splits, desc="Splits", leave=False):
        print(f"\nNumber of {parent_dir} {split} files: {len(data[parent_dir][split])}")
        print(f"First few {parent_dir} {split} paths:")
        for file_path in data[parent_dir][split]:
            source_file = Path(file_path)
            copy_files(source_file, destination_parent_dir, parent_dir_name=parent_dir)