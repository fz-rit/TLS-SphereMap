import json
from pathlib import Path
from pprint import pprint
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


    # Add computed paths to global config
    global_params["output_dir_ls"] = output_dir_ls
    global_params["input_path_ls"] = input_path_ls
    config["global"] = global_params

    return config


# config_path = './input_params/3D_to_2D_config_harvard_forest.json'
# config_path = './input_params/3D_to_2D_config_random_folder.json'
config_path = './input_params/3D_to_2D_config_inlut3d.json'
# config_path = './input_params/3D_to_2D_config_mangrove_roots.json'
CONFIG = load_config_inlut3d(config_path, selected_scans=[2, 3])
# CONFIG = load_config(config_path)
# pprint("🔹 Loaded configuration:"
#        f"\n{CONFIG}")