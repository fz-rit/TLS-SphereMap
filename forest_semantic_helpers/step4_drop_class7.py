import pandas as pd
from pathlib import Path
import argparse


def drop_class7(input_file, output_file):
    """Drop points with class 7 from the input CSV file."""
    df = pd.read_csv(input_file)

    # Check if 'Classification' column exists
    if 'Classification' not in df.columns:
        raise ValueError(f"'Classification' column not found in {input_file}. Please ensure the input file has a 'Classification' column.")

    # Drop points with class 7
    df = df[df['Classification'] != 7]

    # Save the filtered DataFrame to the output file
    df.to_csv(output_file, index=False)
    


input_dir = Path("/home/fzhcis/data/forest_semantic_preprocessed/output/")
output_dir = input_dir.parent / "class7_dropped/"

input_files = list(input_dir.rglob("*_geom_feat.txt"))
input_files.sort(key=lambda x: x.parent.parent.name)  # Sort files for consistent processing
# output_dir.mkdir(parents=True, exist_ok=True)

for input_file in input_files:
    output_dir_perfile = output_dir / input_file.parent.parent.name / "pcd"
    output_file = output_dir_perfile / f"{input_file.stem}_dropclass7.txt"
    # drop_class7(input_file, output_file)
    output_dir_perfile.mkdir(parents=True, exist_ok=True)
    print(f"Processing {input_file} -> {output_dir_perfile}")