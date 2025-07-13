import subprocess
import argparse
import sys
import os
from pathlib import Path

# Add tools directory to Python path for config_loader import
sys.path.append(str(Path(__file__).parent))
from tools.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description='Run 3D to 2D processing pipeline')
    parser.add_argument('--config', '-c', 
                        required=True,
                        help='Path to the configuration JSON file (e.g., input_params/3D_to_2D_config_forestsemantic_rc.json)')
    parser.add_argument('--steps', 
                        nargs='+',
                        default=None,
                        choices=['calc_geom_feature', 'spherical_projection', 'spherical_back_projection', 
                                'convert_color_to_mask', 'attach_segmap_to_points'],
                        help='Specific processing steps to run (default: runs predefined steps)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✅ Successfully loaded configuration from {args.config}")
        print(f"🔹 Dataset: {config['global'].get('dataset', 'Unknown')}")
        print(f"🔹 Processing {len(config['global']['input_path_ls'])} scans")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Set global CONFIG for modules that import it
    import tools.config_loader
    tools.config_loader.CONFIG = config
    
    # Set environment variable for subprocesses
    os.environ['TLS_CONFIG_PATH'] = str(Path(args.config).resolve())

    # Define default processing steps
    default_scripts = [
        "processing.calc_geom_feature",
        "processing.spherical_projection", 
        "processing.spherical_back_projection",
        # "data_annotation.seg_map_tools.convert_color_to_mask", # Only if 2D segmentation map (color) is available.
        # "data_annotation.seg_map_tools.attach_segmap_to_points" # Only if 2D segmentation mask is available.
    ]
    
    # Use custom steps if provided, otherwise use defaults
    if args.steps:
        scripts = [f"processing.{step}" if not step.startswith('processing.') and not step.startswith('data_annotation.') 
                  else step for step in args.steps]
    else:
        scripts = default_scripts

    print(f"\n🚀 Starting pipeline with {len(scripts)} steps...")
    
    for script in scripts:
        print(f"\n🔹 Executing: {script}")
        try:
            result = subprocess.run(["python", "-m", script], check=True)
            print(f"✅ Finished: {script}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {script} failed with return code {e.returncode}")
            break
    else:
        print("\n🎉 All scripts executed successfully!")


if __name__ == "__main__":
    main()
