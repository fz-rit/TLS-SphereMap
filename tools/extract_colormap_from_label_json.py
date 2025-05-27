import json
from pathlib import Path
def extract_colormap(input_file, output_file):
    """
    Extracts the color map from a JSON file and saves it to a new JSON file.

    Args:
        input_file (str): The path to the input JSON file.  Defaults to "labels.json".
        output_file (str): The path to the output JSON file. Defaults to "colormap.json".
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{input_file}'.")
        return

    colormap = {}
    for item in data:
        code = item['code']
        color = item['color']
        # Normalize color values to be between 0 and 1
        normalized_color = tuple(c / 255.0 for c in color)
        colormap[code] = normalized_color

    try:
        with open(output_file, 'w') as f:
            json.dump(colormap, f, indent=4)  # indent for pretty printing
        print(f"Colormap successfully saved to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")

if __name__ == "__main__":
    input_dir = Path("/home/fzhcis/mylab/data/inlut3d")
    input_file= input_dir / "labels.json"
    output_file= input_dir / "colormap.json"
    extract_colormap(input_file, output_file)
