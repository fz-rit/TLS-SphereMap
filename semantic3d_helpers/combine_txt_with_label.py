import numpy as np, laspy, pandas as pd
from pathlib import Path

root_dir = Path("/home/fzhcis/mylab/data/semantic3d/input/domfountain_station3_xyz_intensity_rgb")
file_stem = "domfountain_station3_subsampled_0.01"


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
