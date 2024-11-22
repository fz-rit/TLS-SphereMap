import json
import subprocess
from pathlib import Path
from typing import Union

def print_file_head(file_path: Union[str, Path], num_lines: int = 10) -> None:
    """
    Prints the first few lines of a text file to verify its content.

    Args:
        file_path (str or Path): Path to the text file.
        num_lines (int): Number of lines to print. Default is 10.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\n--- Head of {file_path} ---")
    with file_path.open('r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(line.strip())
    print(f"--- End of Head ---\n")

def run_custom_treeiso(config_file: Union[str, Path]) -> None:
    """
    Executes the CloudCompare `treeiso` command with custom parameters specified in a JSON configuration file.

    Args:
        config_file (str or Path): Path to the JSON configuration file containing input/output paths and parameters.

    Raises:
        FileNotFoundError: If the input file or CloudCompare executable is not found.
        ValueError: If required configuration parameters are missing.
    """
    # Load configuration from JSON file
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with config_file.open('r') as f:
        config = json.load(f)

    # Validate configuration
    required_keys = ['cloudcompare_path', 'input_file', 'root_dir', 'parameters']
    if not all(key in config for key in required_keys):
        raise ValueError(f"Configuration file is missing one or more required keys: {required_keys}")

    cloudcompare_path = Path(config['cloudcompare_path'])
    root_dir = Path(config['root_dir'])
    input_file = root_dir / Path(config['input_file'])
    
    parameters = config['parameters']

    if not cloudcompare_path.exists():
        raise FileNotFoundError(f"CloudCompare executable not found: {cloudcompare_path}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)

    # Print the head of the input file
    print_file_head(input_file)

    # Construct the CloudCompare command
    cmd = [
        str(cloudcompare_path),
        '-SILENT',
        '-AUTO_SAVE', 'OFF',
        '-O', '-SKIP', str(parameters.get('skip', 1)), str(input_file),
        '-C_EXPORT_FMT', 'ASC',
        '-EXT', 'txt',
        '-PREC', str(parameters.get('precision', 8)),
        '-ADD_HEADER',
        '-TREEISO',
        '-LAMBDA1', str(parameters.get('lambda1', 1.0)),
        '-K1', str(parameters.get('k1', 5)),
        '-DECIMATE_RESOLUTION1', str(parameters.get('decimate_resolution1', 0.05)),
        '-LAMBDA2', str(parameters.get('lambda2', 20)),
        '-K2', str(parameters.get('k2', 20)),
        '-MAX_GAP', str(parameters.get('max_gap', 2.0)),
        '-DECIMATE_RESOLUTION2', str(parameters.get('decimate_resolution2', 0.1)),
        '-RHO', str(parameters.get('rho', 0.5)),
        '-VERTICAL_OVERLAP_WEIGHT', str(parameters.get('vertical_overlap_weight', 0.5)),
        '-SAVE_CLOUDS',
        '-LOG_FILE', str(root_dir / parameters.get('log_file', 'cloudcompare_treeiso_log.txt'))
    ]

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("CloudCompare `treeiso` command executed successfully.")
        print(result.stdout.decode())

        # Print the head of the output file if it exists
        output_file = root_dir / input_file.name.replace('.txt', '_MERGED.txt')
        if output_file.exists():
            print_file_head(output_file)
        else:
            print("Expected output file not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing CloudCompare `treeiso`: {e.stderr.decode()}")
        raise

# Example usage
if __name__ == "__main__":
    # Example JSON configuration
    config_json = {
        "cloudcompare_path": "C:\\Program Files\\CloudCompare\\CloudCompare.exe",
        "root_dir": "G:\\My Drive\\projects_with_Jan\\point_cloud_segmentation\\unwrap_outputs\\harvard_forest_33\\33_01",
        "input_file": "33_01_filtered_normaled.txt",
        "parameters": {
            "skip": 1,
            "precision": 8,
            "lambda1": 1.0,
            "k1": 5,
            "decimate_resolution1": 0.05,
            "lambda2": 20,
            "k2": 20,
            "max_gap": 2.0,
            "decimate_resolution2": 0.1,
            "rho": 0.5,
            "vertical_overlap_weight": 0.5,
            "log_file": "cloudcompare_treeiso_log.txt"
        }
    }

    # Save example configuration to file
    config_path = Path("./input_params/isolate_trees_inputs_amiri.json")
    with config_path.open('w') as f:
        json.dump(config_json, f, indent=4)

    # Run the function
    run_custom_treeiso(config_path)
