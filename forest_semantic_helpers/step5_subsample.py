import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint
from forest_semantic_helpers.step0_read_las import observe_df
SUBSAMPLE_RATE = 0.1  # 1% subsample rate

parent_dir = Path("/home/fzhcis/Downloads/ForestSemantic/output")
centered_region_files = list(parent_dir.rglob("region_*_centered*"))

centered_region_files.sort(key=lambda x: x.parent.parent.name + x.stem)
print(f"Found {len(centered_region_files)} recentered region files.")
pprint(centered_region_files)

centered_region_file = centered_region_files[0]

for i, file in enumerate(centered_region_files):
    print(f"Processing {i+1}/{len(centered_region_files)} file: {file.parent.parent.name} - {file.name}")
    pcd_df = pd.read_csv(file)
    # observe_df(pcd_df)
    print(f"Loaded {len(pcd_df):,} points from {file.name}")
    print("Columns:", pcd_df.columns.tolist())

    sample_df = pcd_df.sample(n=int(len(pcd_df) * SUBSAMPLE_RATE), random_state=42).reset_index(drop=True)

    output_dir = parent_dir / "subsampled"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file.parent.parent.name}_{file.stem}_subsampled_{SUBSAMPLE_RATE}.csv"
    sample_df.to_csv(output_path, index=False)
    # observe_df(sample_df)
    print(f"Subsampled points saved: {len(sample_df):,} points to {output_path}")