import json
import os
from pathlib import Path
from pprint import pprint
import numpy as np
from tools.pcd_utils import create_dir_if_not_exists


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

def _add_common_config_params(config):
    """Add common computed parameters to config."""
    global_params = config["global"]
    
    # Calculate canvas size
    v_fov = global_params['v_fov']
    h_fov = global_params['h_fov']
    canvas_size = (
        int((v_fov[1] - v_fov[0]) / global_params['v_ang_res_deg']),
        int((h_fov[1] - h_fov[0]) / global_params['h_ang_res_deg'])
    )
    global_params["canvas_size"] = canvas_size
    config["global"] = global_params
    return config


def load_config_mangrove(config):
    """Load the JSON config and dynamically generate paths for mangrove dataset."""
    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_folder = global_params["input_folder"]
    input_suffix = global_params.get("input_suffix", ".txt")
    input_folder_parent = global_params["input_folder_parent"]
    selected_scans = global_params['selected_scans']
    output_root_dir = output_base_dir / input_folder_parent / input_folder
    create_dir_if_not_exists(output_root_dir, ask_user=False)
    output_sub_folders = [p for p in output_root_dir.iterdir() if p.is_dir() and p.name != "forest"]
    output_sub_folders.sort()
    output_sub_folders = output_sub_folders[selected_scans[0]:selected_scans[1]]
    output_sub_folder_names = [p.name for p in output_sub_folders]
    print(f"🔹 Found output sub-folders: {output_sub_folder_names}")
    output_dir_ls = [p / "outputs" for p in output_sub_folders]
    
    input_path_ls = []
    for output_sub_folder_name in output_sub_folder_names:
        input_path = input_base_dir / input_folder_parent / input_folder / f"{output_sub_folder_name}{input_suffix}"
        if not input_path.exists():
            raise FileNotFoundError(f"❗ Input file {input_path} does not exist.")
        input_path_ls.append(input_path)

    # Add computed paths to global config
    color_map = get_color_map(input_base_dir/input_folder_parent)
    global_params["color_map"] = color_map
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls

    return _add_common_config_params(config)


def load_config_forestsemantic(config):
    """Load the JSON config and dynamically generate paths for forest semantic dataset."""
    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_suffix = global_params.get("input_suffix", ".csv")
    selected_scans = global_params['selected_scans']
    create_dir_if_not_exists(output_base_dir, ask_user=False)

    input_path_ls = list(input_base_dir.glob(f"plot*_centered_subsample_scan*{input_suffix}"))
    if not input_path_ls:
        raise FileNotFoundError(f"❗ No input files found in {input_base_dir} with suffix {input_suffix}.")
    input_path_ls.sort(key=lambda x: x.stem)
    output_dir_ls = []
    for input_path in input_path_ls[selected_scans[0]:selected_scans[1]]:
        output_dir = output_base_dir / input_path.stem
        create_dir_if_not_exists(output_dir, ask_user=False)
        output_dir_ls.append(output_dir)

    # Add computed paths to global config
    color_map = get_color_map(input_base_dir)
    global_params["color_map"] = color_map
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls

    return _add_common_config_params(config)



def load_config_inlut3d(config):
    """Load the JSON config and dynamically generate paths for INLUT3D dataset."""
    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_suffix = global_params.get("input_suffix", ".las")
    selected_scans = global_params['selected_scans']
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

    color_map = get_color_map(input_base_dir)
    global_params["color_map"] = color_map
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls

    return _add_common_config_params(config)


def load_config_semantic3d(config):
    """Load the JSON config and dynamically generate paths for Semantic3D dataset."""
    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_suffix = global_params["input_suffix"]
    selected_scans = global_params['selected_scans']
    input_folders = list(input_base_dir.iterdir())
    input_folders = [p for p in input_folders if p.is_dir()]
    input_folders.sort()
    output_dir_ls = []
    input_path_ls = []
    for p in input_folders[selected_scans[0]:selected_scans[1]]:
        input_path = next(p.glob(f"*{input_suffix}"), None)
        if not input_path.exists():
            raise FileNotFoundError(f"❗ Input file {input_path} does not exist.")
        input_path_ls.append(input_path)
        output_dir = output_base_dir / p.name
        create_dir_if_not_exists(output_dir, ask_user=False)
        output_dir_ls.append(output_dir)

    color_map = get_color_map(input_base_dir)
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls
    global_params["color_map"] = color_map

    return _add_common_config_params(config)


def load_config(config_path):
    """Load configuration from JSON file and apply dataset-specific processing."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"❗ Config file {config_path} does not exist.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Determine dataset type and apply appropriate loader
    dataset = config["global"].get("dataset", "").lower()
    
    if dataset == "mangrove" or "mangrove" in str(config_path).lower():
        return load_config_mangrove(config)
    elif dataset == "forestsemantic" or "forestsemantic" in str(config_path).lower():
        return load_config_forestsemantic(config)
    elif dataset == "inlut3d" or "inlut3d" in str(config_path).lower():
        return load_config_inlut3d(config)
    elif dataset == "semantic3d" or "semantic3d" in str(config_path).lower():
        return load_config_semantic3d(config)
    else:
        # Try to infer from filename if dataset field is not set
        if "mangrove" in str(config_path).lower():
            return load_config_mangrove(config)
        elif "forest" in str(config_path).lower():
            return load_config_forestsemantic(config)
        elif "inlut" in str(config_path).lower():
            return load_config_inlut3d(config)
        elif "semantic3d" in str(config_path).lower():
            return load_config_semantic3d(config)
        else:
            # Default to forest semantic if cannot determine
            print(f"⚠️ Warning: Could not determine dataset type from {config_path}. Using forestsemantic loader.")
            return load_config_forestsemantic(config)


# Global CONFIG variable for backward compatibility
CONFIG = None

# Auto-load config if environment variable is set
def _auto_load_config():
    global CONFIG
    if CONFIG is None:
        config_path = os.environ.get('TLS_CONFIG_PATH')
        if config_path and Path(config_path).exists():
            CONFIG = load_config(config_path)
            print(f"🔹 Auto-loaded configuration from environment: {config_path}")
    return CONFIG

# Try to auto-load on import
_auto_load_config()