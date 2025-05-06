import argparse
import numpy as np
import pandas as pd
import json
from PIL import Image
import sys
from pathlib import Path
from plyfile import PlyData, PlyElement
# Add parent directory to sys.path
current_file = Path(__file__).resolve()
current_file_dir = current_file.parent


from tools.preprocess_point_cloud import map_angle_to_pixel
from tools.config_loader import CONFIG


def load_label_maps(dataset_name):
    """Load dataset-specific colormap and class names from label_maps.json."""
    with open(current_file_dir / "label_maps.json", "r") as file:
        label_maps = json.load(file)["DATASETS"]

    if dataset_name not in label_maps:
        raise ValueError(f"Dataset '{dataset_name}' not found in label_maps.json.")

    # Convert string keys to tuple format for colors
    color_to_index = {tuple(map(int, k.split(","))): v for k, v in label_maps[dataset_name]["COLOR_TO_INDEX"].items()}
    
    # Convert class index names
    class_names = {int(k): v for k, v in label_maps[dataset_name]["CLASS_NAMES"].items()}
    
    # Reverse lookup: Index → RGB color
    index_to_color = {v: k for k, v in color_to_index.items()}

    return color_to_index, class_names, index_to_color


def attach_segmentation_to_points():
    """Attach segmentation map class IDs and colors to the point cloud using parameters from JSON."""
    params = CONFIG["attach_segmap_to_points"]
    root_dir = CONFIG["global"]["output_dir"] / 'pcd'
    # input_file_stem = CONFIG["global"]["input_file_stem"]
    # point_cloud_file = root_dir / f"{input_file_stem}_filtered_normaled_curvature_0.06_roughness_0.06.txt"
    point_cloud_file = next(root_dir.glob(f"*_ncr_*"), None)

    segmap_file = root_dir / params["segmap"]
    dataset_name = params["dataset"]
    output_formats = params["output_formats"]

    # Load point cloud
    pc_df = pd.read_csv(point_cloud_file, sep=',')
    
    # Rename "Return Number" to "return_number" to avoid space issues
    pc_df.rename(columns={"Return Number": "return_number"}, inplace=True)
    pc_df.rename(columns={"range1metres": "range"}, inplace=True)

    azimuth, elevation = pc_df['azimuth'], pc_df['elevation']

    # Map angles to segmentation map pixels
    x_pix, y_pix = map_angle_to_pixel(azimuth, elevation)
    pc_df['x_pix'] = x_pix.astype(int)
    pc_df['y_pix'] = y_pix.astype(int)

    # Load segmentation map
    segmap = np.array(Image.open(segmap_file))  # Shape: (H, W)

    # Load label maps
    _, class_names, index_to_color = load_label_maps(dataset_name)

    # Attach class ID and RGB color to points
    class_ids = []
    colors = []
    height, width = segmap.shape  # Size of segmentation image

    for x, y in zip(pc_df['x_pix'], pc_df['y_pix']):
        # Ensure pixel indices are within image bounds
        if 0 <= x < width and 0 <= y < height:
            class_id = segmap[y, x]  # Get class ID from segmentation map
        else:
            class_id = 0  # Default to Void if out of bounds

        class_ids.append(class_id)
        colors.append(index_to_color.get(class_id, (0, 0, 0)))  # Default to black if not found

    pc_df['class_id'] = class_ids
    pc_df[['r', 'g', 'b']] = pd.DataFrame(colors, index=pc_df.index)

    # Rescale Intensity values to 0-65535 to ensure compatibility with las format, astype uint16
    if pc_df['Intensity'].max() <= 1.0:
        pc_df['Intensity'] = (pc_df['Intensity'] * 65535).astype(np.uint16)


    # Drop extra columns used for processing
    pc_df.drop(columns=['x_pix', 'y_pix'], inplace=True)

    # Reorder columns
    pc_df = pc_df[['X', 'Y', 'Z', 'Intensity', 'return_number', 'azimuth', 'elevation', 'zenith',
                   'range', 'nx', 'ny', 'nz', 'curvature', 'roughness', 'class_id', 'r', 'g', 'b']]

    # Save as various output formats
    for output_format in output_formats:
        file_str = str(point_cloud_file.stem).split("_filtered")[0]
        output_file = root_dir / (f"{file_str}_seg" + output_format)
        if output_format == ".csv":
            pc_df.to_csv(output_file, index=False)
        elif output_format == ".ply":
            # Convert DataFrame to structured array
            dtype_list = [
                ('X', 'f4'), ('Y', 'f4'), ('Z', 'f4'),
                ('Intensity', 'u2'), # ensure las compatibility
                ('return_number', 'u1'),
                ('azimuth', 'f4'), ('elevation', 'f4'), ('zenith', 'f4'),
                ('range', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('curvature', 'f4'), ('roughness', 'f4'), ('class_id', 'u1'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
            ]

            data_np = np.array([tuple(row) for row in pc_df.to_numpy()], dtype=dtype_list)

            # Create PLY element
            vertex = PlyElement.describe(data_np, 'vertex')

            # Write to file
            ply_data = PlyData([vertex], text=False)
            ply_data.write(output_file) 
        else:
            raise ValueError(f"!!Output format '{output_format}' not supported.") 

        print(f"✅ Class IDs and colors attached to point cloud successfully. \nOutput file saved to: {output_file}")
    print("Class details:")
    for class_id, class_name in class_names.items():
        print(f"Class ID {class_id}: {class_name}")

if __name__ == "__main__":
    attach_segmentation_to_points()
