import json
from pathlib import Path
from pprint import pprint
import numpy as np
from tools.pcd_utils import create_dir_if_not_exists
# from pcd_utils import create_dir_if_not_exists



def load_config(config_path):
    """Load the JSON config and dynamically generate paths."""
    with open(config_path, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_folder = global_params["input_folder"]
    input_suffix = global_params.get("input_suffix", ".txt")  # Default to .txt if not specified
    input_folder_parent = global_params["input_folder_parent"]

    output_root_dir  = output_base_dir / input_folder_parent / input_folder
    create_dir_if_not_exists(output_root_dir, ask_user=False)
    output_sub_folder = [p for p in output_root_dir.iterdir() if p.is_dir() and p.name != "forest"]
    output_sub_folder.sort()
    output_sub_folder_names = [p.name for p in output_sub_folder]
    print(f"🔹 Found output sub-folders: {output_sub_folder_names}")
    output_dir_ls = [p / "outputs" for p in output_sub_folder]
    
    input_path_ls = []
    for output_sub_folder_name in output_sub_folder_names:
        input_path = input_base_dir / input_folder_parent / input_folder / f"{output_sub_folder_name}{input_suffix}"
        if not input_path.exists():
            raise FileNotFoundError(f"❗ Input file {input_path} does not exist.")
        input_path_ls.append(input_path)

    # Add computed paths to global config
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls
    config["global"] = global_params

    return config

def get_color_map(input_base_dir):
    label_file = input_base_dir / 'labels.json'
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    color_map = {label_dict['code']:label_dict["color"] for label_dict in label_json}
    color_map = {k: v for k, v in color_map.items() if k <18}

    color_arr = np.array(list(color_map.values()))

    scale = 1/255 if color_arr.max() > 1 else 1
    color_arr = color_arr * scale
    color_list = [tuple(color) for color in color_arr]
    color_map = {str(k): tuple(color) for k, color in zip(color_map.keys(), color_list)}

    return color_map

def load_config_inlut3d(config_path, selected_scans = [1,30]):
    """Load the JSON config and dynamically generate paths."""
    with open(config_path, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_suffix = global_params.get("input_suffix", ".las")  # Default to .txt if not specified

    input_folders = list(input_base_dir.iterdir())
    input_folders = [p for p in input_folders if p.is_dir()]
    input_folders_sort = sorted(input_folders, key=lambda p: int(p.name.split('_')[1]))
    output_dir_ls = []
    input_path_ls = []
    for p in input_folders_sort[selected_scans[0]:selected_scans[1]]:
        input_path = next(p.glob(f"*{input_suffix}"), None)
        if not input_path.exists():
            raise FileNotFoundError(f"❗ Input file {input_path} does not exist.")
        input_path_ls.append(input_path)
        output_dir = output_base_dir / p.name
        create_dir_if_not_exists(output_dir, ask_user=False)
        output_dir_ls.append(output_dir)

    # color_map_file = input_base_dir / "colormap.json"
    # with open(color_map_file, 'r') as f:
    #         color_map = json.load(f)
    # Add computed paths to global config
    color_map = get_color_map(input_base_dir)
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls
    global_params["color_map"] = color_map
    config["global"] = global_params

    return config


def load_config_semantic3d(config_path, selected_scans = [1,30]):
    """Load the JSON config and dynamically generate paths."""
    with open(config_path, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_suffix = global_params["input_suffix"]

    input_folders = list(input_base_dir.iterdir())
    input_folders = [p for p in input_folders if p.is_dir() and "pcd" in p.name]
    input_folders_sort = sorted(input_folders)
    output_dir_ls = []
    input_path_ls = []
    for p in input_folders_sort[selected_scans[0]:selected_scans[1]]:
        print(f"🔹 Found input folder: {p.name}")
        input_path = next(p.glob(f"*{input_suffix}"), None)
        if not input_path.exists():
            raise FileNotFoundError(f"❗ Input file {input_path} does not exist.")
        input_path_ls.append(input_path)
        output_dir = output_base_dir / p.name
        create_dir_if_not_exists(output_dir, ask_user=False)
        output_dir_ls.append(output_dir)

    # color_map_file = input_base_dir / "colormap.json"
    # with open(color_map_file, 'r') as f:
    #         color_map = json.load(f)
    # Add computed paths to global config
    color_map = get_color_map(input_base_dir)
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls
    global_params["color_map"] = color_map
    config["global"] = global_params

    return config

# config_path = './input_params/3D_to_2D_config_harvard_forest.json'
config_path = './input_params/3D_to_2D_config_semantic3d.json'
# config_path = './input_params/3D_to_2D_config_mangrove_roots.json'
# CONFIG = load_config_inlut3d(config_path, selected_scans=[122, 321])
# CONFIG = load_config(config_path)
CONFIG = load_config_semantic3d(config_path, selected_scans=[0, 1])
pprint("🔹 Loaded configuration:"
       f"\n{CONFIG}")