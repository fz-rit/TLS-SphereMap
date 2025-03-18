import json
from pathlib import Path
CONFIG_PATH = './input_params/3D_to_2D_config_mangrove_roots.json'
# CONFIG_PATH = './input_params/3D_to_2D_config_harvard_forest.json'

def load_config():
    """Load the JSON config and dynamically generate paths."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    
    input_file_stem = global_params["input_file_stem"]
    input_folder = global_params["input_folder"]
    input_folder_parent = global_params["input_folder_parent"]
    output_base_dir = Path(global_params["output_base_dir"])
    input_base_dir = Path(global_params["input_base_dir"])


    # Compute dynamic paths
    output_dir = output_base_dir / input_folder_parent / input_folder / input_file_stem / "outputs"
    input_suffix = ".txt" if "mangrove" in CONFIG_PATH else ".las"
    input_path = input_base_dir / input_folder_parent / input_folder / f"{input_file_stem}{input_suffix}"

    # Add computed paths to global config
    global_params["output_dir"] = output_dir
    global_params["input_path"] = input_path
    config["global"] = global_params

    return config

# Load the configuration once so other scripts can access it
CONFIG = load_config()
