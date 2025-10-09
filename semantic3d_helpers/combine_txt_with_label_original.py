import numpy as np, laspy, pandas as pd
from pathlib import Path

# root_dir = Path("/home/fzhcis/mylab/data/semantic3d/input/test")
# file_stem = "domfountain_station3_xyz_intensity_rgb"

def attach_labels_to_points(txt_file: Path, labels_file: Path, las_dir:Path):
    """Attach labels from .labels file to points in .txt file and save as .las."""

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

    las.write(las_dir / f'{txt_file.stem}_with_labels.las')


root_dir = Path("/home/fzhcis/data/semantic3d_reduced8/semantic3d/pcd")
pcd_dir = root_dir / "train"
label_dir = root_dir / "sem8_labels_training"
las_dir = root_dir / "las_train_with_labels"

pcd_files = list(pcd_dir.glob("*.txt"))
label_files = list(label_dir.glob("*.labels"))

pcd_files.sort()
label_files.sort()

assert len(pcd_files) > 0, "No .txt files found in the directory."
assert len(label_files) > 0, "No .labels files found in the directory."

if len(pcd_files) != len(label_files):
    raise ValueError(f"Number of .txt files ({len(pcd_files)}) does not match number of .labels files ({len(label_files)}).")

for txt_file, labels_file in zip(pcd_files[7:], label_files[7:]):
    if txt_file.stem != labels_file.stem:
        raise ValueError(f"File names do not match: {txt_file.name} and {labels_file.name}")
    print(f"Processing {txt_file.stem} .txt+.labels ...")
    attach_labels_to_points(txt_file, labels_file, las_dir)



# # ------Read the las file and convert to pandas DataFrame--------
# las_file = laspy.read(root_dir / f'{file_stem}_with_labels.las')
# data = {
#     'X': np.array(las_file.x),
#     'Y': np.array(las_file.y),
#     'Z': np.array(las_file.z),
#     'Intensity': np.array(las_file.intensity),
#     'r': np.array(las_file.red),
#     'g': np.array(las_file.green),
#     'b': np.array(las_file.blue),
#     'Classification': np.array(las_file.classification)
# }
# df = pd.DataFrame(data)


# df['Classification'] = df['Classification'].astype(np.uint8)
# print(df.head())
# print(f"First 10 classifications: {df['Classification'].iloc[:10].values}")
# print(f"Unique classifications: {df['Classification'].unique()}")
# print(f"Shape of DataFrame: {df.shape}")
# print(f"Each point has classification: {df['Classification'].iloc[0]}, {df['Classification'].iloc[1]}, {df['Classification'].iloc[2]}")


# def observe_df(pcd_df: pd.DataFrame) -> None:
#     """Observe the DataFrame structure and basic statistics."""
#     for col in pcd_df.columns:
#         print(f"\n{col}:")
#         print(f"  Data type: {pcd_df[col].dtype}")
#         print(f"  Shape: {pcd_df[col].shape}")
#         try:
#             if pcd_df[col].dtype == 'object':
#                 print(f"  Sample values: {pcd_df[col].iloc[:5].tolist()}")
#             else:
#                 print(f"  Range: {pcd_df[col].min()} - {pcd_df[col].max()}")
#                 print(f"  Mean: {pcd_df[col].mean():.2f}")
#         except Exception as e:
#             print(f"  Error computing stats: {e}")
#             print(f"  Sample values: {pcd_df[col].iloc[:5].tolist()}")

# observe_df(df)