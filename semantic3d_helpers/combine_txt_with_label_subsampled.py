import numpy as np, laspy, pandas as pd
from pathlib import Path


def attach_labels_to_points(individual_dir: Path, las_dir:Path):
    """Attach labels from .labels file to points in .txt file and save as .las."""

    txt_file = individual_dir / f"_subsampled_voxel_subsampled_0.01.txt"
    labels_file = individual_dir / f"_subsampled_voxel_subsampled_0.01.labels"

    if not txt_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Input files {txt_file} or {labels_file} do not exist.")

    xyzirgb = np.loadtxt(txt_file, usecols=(0,1,2,3,4,5,6)) # x,y,z,intensity,r,g,b
    labels = np.loadtxt(labels_file, dtype=np.uint8)
    las = laspy.create(point_format=3, file_version='1.4')
    las.x, las.y, las.z = xyzirgb[:,:3].T
    las.intensity = xyzirgb[:,3]
    las.red = xyzirgb[:,4]
    las.green = xyzirgb[:,5]
    las.blue = xyzirgb[:,6]
    las.classification = labels
    print(f"Before dropping class 0, unique labels: {np.unique(las.classification)}; shape: {las.classification.shape}")
    # Drop Class 0 as it is not used in Semantic3D:
    # ref: http://www.semantic3d.net/view_dbase.php?chl=1
    las = las[las.classification != 0]
    print(f"After dropping class 0, unique labels: {np.unique(las.classification)}; shape: {las.classification.shape}")

    las.write(las_dir / f'{individual_dir.name}_with_labels.las')


root_dir = Path("/home/fzhcis/data/semantic3d_reduced8/subsampled")
las_dir = root_dir.parent / "subsampled_las_train_with_labels"
las_dir.mkdir(parents=True, exist_ok=True)
individual_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
individual_dirs.sort(key=lambda folder: folder.name)
for individual_dir in individual_dirs:

    print(f"Processing {individual_dir.name} .txt+.labels ...")
    attach_labels_to_points(individual_dir, las_dir)

