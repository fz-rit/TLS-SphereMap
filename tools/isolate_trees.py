import json
import subprocess
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd

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

def reformat_outputfile(root_dir: Union[str, Path], log_file: Union[str, Path], keep_intermid_file:bool=False) -> None:
    """
    Reformat the output file from CloudCompare `treeiso` command to a more readable format.

    Args:
        root_dir (str or Path): Path to the input file.
        output_file (str or Path): Path to the output file.

    Raises:
        FileNotFoundError: If the input file is not found.
    """
    intermediate_file = next(root_dir.glob("*filtered_normaled_*-*-*.txt"), None)
    if intermediate_file is None:
        raise FileNotFoundError(f"No file matching pattern '*filtered_normaled_*-*-*.txt' found in {root_dir}")
    output_file = root_dir / (intermediate_file.name.split('_filtered_normaled_')[0] + '_filtered_normaled_treeiso.txt')
    if not intermediate_file.exists():
        raise FileNotFoundError(f"Input file not found: {intermediate_file}")

    # Read the input file and extract the header line
    with intermediate_file.open('r') as f:
        header = f.readline().strip()
        data = np.loadtxt(f, delimiter=',')

    # Modify the header line
    header = header.replace('//X', 'X')
    header = header.replace('Scalar field #2', 'Return Number')
    header = header.replace('Scalar field #3', 'azimuth')
    header = header.replace('Scalar field #4', 'zenith')
    header = header.replace('Scalar field #5', 'range1metres')
    header = header.replace('Scalar field #6', 'x_pix')
    header = header.replace('Scalar field #7', 'y_pix')
    header = header.replace('Scalar field', 'Intensity')
    header = header.replace('final_segs', 'treeiso_label')
    header = header.replace('Nx', 'nx')
    header = header.replace('Ny', 'ny')
    header = header.replace('Nz', 'nz')

    columns = header.split(',')
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({ # the default type for float is 'float64', which consumes more space than necessary.
                    'X': 'float32',
                    'Y': 'float32',
                    'Z': 'float32',
                    'Intensity': 'uint16',
                    'Return Number': 'uint8',
                    'azimuth': 'float32',
                    'zenith': 'float32',
                    'range1metres': 'float32',
                    'x_pix': 'uint16',
                    'y_pix': 'uint16',
                    'treeiso_label': 'uint16', # upto 65535
                    'nx': 'float32',
                    'ny': 'float32',
                    'nz': 'float32'
                })
    
    # Drop the columns named "init_segs" and "intermediate_segs"
    df.drop(columns=["init_segs", "intermediate_segs"], inplace=True, errors='ignore')

    # Write the reformatted data to the output file
    df.to_csv(output_file, index=False, float_format='%g')
    print(f"Reformatted data written to {output_file}")
    

    if not keep_intermid_file:
        original_file = root_dir / (intermediate_file.name.split('_filtered_normaled_')[0] + '_filtered_normaled.txt')
        original_file.unlink()
        intermediate_file.unlink()
        log_file.unlink()
        print(f"Intermediate file {intermediate_file} deleted.")
        print(f"Original file {original_file} deleted.")
        print(f"Log file {log_file} deleted.")
        

    return output_file
    

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
    keep_intermid_file = config.get('keep_intermid_file', False)
    parameters = config['parameters']
    log_file = root_dir / parameters.get('log_file', 'cloudcompare_treeiso_log.txt')

    if not cloudcompare_path.exists():
        raise FileNotFoundError(f"CloudCompare executable not found: {cloudcompare_path}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)

    # Print the head of the input file
    print("Input file head:")
    print_file_head(input_file)

    # Construct the CloudCompare command
    cc_treeiso_cmd = [
        str(cloudcompare_path),
        '-SILENT',
        '-AUTO_SAVE', 'OFF',
        '-LOG_FILE', str(log_file),
        '-O', '-SKIP', str(parameters.get('skip', 1)), str(input_file),
        '-C_EXPORT_FMT', 'ASC',
        '-EXT', 'txt',
        '-SEP', 'COMMA',
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
        '-SAVE_CLOUDS'
    ]

    print(f"Executing CloudCompare `treeiso` command")
    command_print = [f'"{item}"' if ' ' in item else item for item in cc_treeiso_cmd]
    print(f"CloudCompare Command used in terminal: {' '.join(command_print)}")


    # Run the command
    try:
        result = subprocess.run(cc_treeiso_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("CloudCompare `treeiso` command executed successfully.")
        print(result.stdout.decode())

        # Print the head of the output file if it exists
        output_file = reformat_outputfile(root_dir, log_file, keep_intermid_file=keep_intermid_file)
        if output_file.exists():
            print_file_head(output_file)
        else:
            print("Expected output file not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing CloudCompare `treeiso`: {e.stderr.decode()}")
        raise

# Example usage
if __name__ == "__main__":
    # # Example JSON configuration
    # config_json = {
    #     "cloudcompare_path": "C:\\Program Files\\CloudCompare\\CloudCompare.exe",
    #     "root_dir": "G:\\My Drive\\projects_with_Jan\\point_cloud_segmentation\\unwrap_outputs\\harvard_forest_33\\33_01",
    #     "input_file": "33_01_filtered_normaled.txt",
    #     "keep_intermid_file": False,
    #     "parameters": {
    #         "skip": 1,
    #         "precision": 8,
    #         "lambda1": 1.0,
    #         "k1": 5,
    #         "decimate_resolution1": 0.05,
    #         "lambda2": 20,
    #         "k2": 20,
    #         "max_gap": 2.0,
    #         "decimate_resolution2": 0.1,
    #         "rho": 0.5,
    #         "vertical_overlap_weight": 0.5,
    #         "log_file": "cloudcompare_treeiso_log.txt"
    #     }
    # }

    # # Save example configuration to file
    # config_path = Path("./input_params/isolate_trees_inputs_amiri.json")
    # with config_path.open('w') as f:
    #     json.dump(config_json, f, indent=4)

    # Run the function
    config_path = Path("./input_params/isolate_trees_inputs_amiri.json")
    run_custom_treeiso(config_path)
