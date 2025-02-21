import json

CONFIG_PATH = './input_params/3D_to_2D_config_mangrove_roots.json'

def load_config():
    """Load the JSON config and dynamically generate paths."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    global_params = config["global"]
    
    # Compute dynamic paths
    input_file_stem = global_params["input_file_stem"]
    input_folder = global_params["input_folder"]
    
    output_dir = f"/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/{input_folder}/{input_file_stem}/outputs"
    
    # Add computed paths to global config
    global_params["output_dir"] = output_dir

    config["global"] = global_params

    return config

# Load the configuration once so other scripts can access it
CONFIG = load_config()
