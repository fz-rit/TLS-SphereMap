import json
from pathlib import Path
from pprint import pprint
CONFIG_PATH = './input_params/3D_to_2D_config_mangrove_roots.json'
# CONFIG_PATH = './input_params/3D_to_2D_config_harvard_forest.json'
# CONFIG_PATH = './input_params/3D_to_2D_config_random_folder.json'

def load_config():
    """Load the JSON config and dynamically generate paths."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])
    input_folder = global_params["input_folder"]
    input_suffix = global_params.get("input_suffix", ".txt")  # Default to .txt if not specified
    input_folder_parent = global_params["input_folder_parent"]

    output_root_dir  = output_base_dir / input_folder_parent / input_folder
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

# Load the configuration once so other scripts can access it
CONFIG = load_config()
# pprint("🔹 Loaded configuration:"
#        f"\n{CONFIG}")