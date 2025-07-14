import numpy as np, laspy, pandas as pd
from pathlib import Path

root_dir = Path("/home/fzhcis/mylab/data/semantic3d/input/test")
file_stem = "domfountain_station3_xyz_intensity_rgb"


# -----Load the .txt file and .labels file and create a .las file from them.------
xyzirgb = np.loadtxt(root_dir / f'{file_stem}.txt', usecols=(0,1,2,3,4,5,6)) # x,y,z,intensity,r,g,b
labels = np.loadtxt(root_dir / f'{file_stem}.labels', dtype=np.uint8)
las = laspy.create(point_format=3, file_version='1.4')
las.x, las.y, las.z = xyzirgb[:,:3].T
las.intensity = xyzirgb[:,3]
las.red = xyzirgb[:,4]
las.green = xyzirgb[:,5]
las.blue = xyzirgb[:,6]
las.classification = labels
# Drop Class 0 as it is not used in Semantic3D:
# ref: http://www.semantic3d.net/view_dbase.php?chl=1
las = las[las.classification != 0]
las.write(root_dir / f'{file_stem}_with_labels.las')


# ------Read the las file and convert to pandas DataFrame--------
las_file = laspy.read(root_dir / f'{file_stem}_with_labels.las')
data = {
    'X': np.array(las_file.x),
    'Y': np.array(las_file.y),
    'Z': np.array(las_file.z),
    'Intensity': np.array(las_file.intensity),
    'r': np.array(las_file.red),
    'g': np.array(las_file.green),
    'b': np.array(las_file.blue),
    'Classification': np.array(las_file.classification)
}
df = pd.DataFrame(data)


df['Classification'] = df['Classification'].astype(np.uint8)
print(df.head())
print(f"First 10 classifications: {df['Classification'].iloc[:10].values}")
print(f"Unique classifications: {df['Classification'].unique()}")
print(f"Shape of DataFrame: {df.shape}")
print(f"Each point has classification: {df['Classification'].iloc[0]}, {df['Classification'].iloc[1]}, {df['Classification'].iloc[2]}")


def observe_df(pcd_df: pd.DataFrame) -> None:
    """Observe the DataFrame structure and basic statistics."""
    for col in pcd_df.columns:
        print(f"\n{col}:")
        print(f"  Data type: {pcd_df[col].dtype}")
        print(f"  Shape: {pcd_df[col].shape}")
        try:
            if pcd_df[col].dtype == 'object':
                print(f"  Sample values: {pcd_df[col].iloc[:5].tolist()}")
            else:
                print(f"  Range: {pcd_df[col].min()} - {pcd_df[col].max()}")
                print(f"  Mean: {pcd_df[col].mean():.2f}")
        except Exception as e:
            print(f"  Error computing stats: {e}")
            print(f"  Sample values: {pcd_df[col].iloc[:5].tolist()}")

observe_df(df)