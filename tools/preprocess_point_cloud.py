import pandas as pd
from pathlib import Path
import laspy
import numpy as np

# HORIZONTAL_FOV = 360.0
# # ======For TLS data======
# VERTICAL_FOV = 135.0
# VERTICAL_ANGLE_RESOLUTION = 0.25
# HORIZONTAL_ANGLE_RESOLUTION = 0.25

# ===For SemanticKitti data (Velodyne-HDL-64)===
# VERTICAL_FOV = 26.8
# VERTICAL_ANGLE_RESOLUTION = 0.4188
# HORIZONTAL_ANGLE_RESOLUTION = 0.08

# # ===For InLUT3D data===
# VERTICAL_FOV = 150.0
# VERTICAL_ANGLE_RESOLUTION = 0.0229183
# HORIZONTAL_ANGLE_RESOLUTION = 0.0229183


def convert_SemanticKitti_bin_to_pcd(bin_file: Path, pcd_file: Path):
    """
    Convert SemanticKitti bin file to pcd file.
    Parameters:
    bin_file (Path): The path to the SemanticKitti bin file.
    pcd_file (Path): The path to save the pcd file.
    """
    points = np.fromfile(bin_file, dtype=np.float32)
    points = points.reshape((-1, 4))
    print(points.shape)

    # ####Prepare the PCD header (by ChatGPT)####
    header = f"""# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z intensity
    SIZE 4 4 4 4
    TYPE F F F F
    COUNT 1 1 1 1
    WIDTH {points.shape[0]}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {points.shape[0]}
    DATA ascii
    """
    ############################################

    # Save the point cloud to a .pcd file
    with open(pcd_file, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")

    print(f"Converted {bin_file} to {pcd_file}.")
    return points

def calculate_zenith_angles(las_x: np.ndarray, las_y: np.ndarray, las_z: np.ndarray) -> np.ndarray:
    """
    Calculate zenith angles for multiple points given x, y, z components.

    Args:
        las_x (np.ndarray): Array of x-coordinates.
        las_y (np.ndarray): Array of y-coordinates.
        las_z (np.ndarray): Array of z-coordinates.

    Returns:
        np.ndarray: Zenith angles in degrees for all points, with 0 assigned where magnitudes are 0.
    """

    # Stack the input vectors into a single array
    points = np.column_stack((las_x, las_y, las_z))

    # Calculate the magnitudes of the vectors
    magnitudes = np.linalg.norm(points, axis=1)

    # Avoid division by zero by assigning 0 to zenith_angles where magnitudes are 0
    with np.errstate(divide='ignore', invalid='ignore'):
        zenith_angles = np.arccos(np.divide(points[:, 2], magnitudes, where=magnitudes != 0)) * 180 / np.pi
        zenith_angles[magnitudes == 0] = 0

    return zenith_angles # range from 0 to 180



def map_angle_to_pixel(azimuth: np.ndarray, 
                       elevation: np.ndarray, 
                       canvas_size: tuple[int, int],
                       angular_res: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Map azimuth and elevation angles to pixel coordinates on the unwrapped image.
    
    Parameters:
        azimuth (np.ndarray): Azimuth angles in degrees.
        elevation (np.ndarray): Elevation angles in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y pixel coordinates

    """

    CANVAS_HEIGHT, CANVAS_WIDTH = canvas_size
    VERTICAL_ANGLE_RESOLUTION, HORIZONTAL_ANGLE_RESOLUTION = angular_res
    x_pix = (azimuth / HORIZONTAL_ANGLE_RESOLUTION).astype(int)

    y_pix = CANVAS_HEIGHT - ((elevation - elevation.min()) / VERTICAL_ANGLE_RESOLUTION).astype(int)

    x_pix = x_pix.clip(0, CANVAS_WIDTH - 1)
    y_pix = y_pix.clip(0, CANVAS_HEIGHT - 1)
    return x_pix, y_pix

def read_pts_file(file_path):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # First line is the number of points, skip it
    data_lines = lines[1:]

    # Convert to DataFrame
    from io import StringIO
    data_str = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep='\s+', header=None)
    df.columns = ["X", "Y", "Z", "r", "g", "b", "class_id", "instance_id"]

    return df

def add_angle_range_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add azimuth and zenith angles to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The point cloud data DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with added azimuth and zenith angles.
    """
    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        raise ValueError("DataFrame must contain 'X', 'Y', and 'Z' columns to calculate angles.")
    if all(col in df.columns for col in ['azimuth', 'zenith', 'elevation', 'rangemeter']):
        raise ValueError("DataFrame already contains 'azimuth' and 'zenith' columns.")

    df['azimuth'] = np.arctan2(df['Y'].values, df['X'].values) * 180 / np.pi
    df['azimuth'] = np.where(df['azimuth'] < 0, 360 + df['azimuth'], df['azimuth'])
    zenith_angles = calculate_zenith_angles(df['X'].values, df['Y'].values, df['Z'].values)
    df['elevation'] = 90 - zenith_angles
    df['zenith'] = zenith_angles
    df['rangemeter'] = (df['X'] ** 2 + df['Y'] ** 2 + df['Z'] ** 2) ** 0.5
    return df

def read_raw_point_cloud(filename: Path, dataset_name:str="MANGROVE", flip_mangrove:bool=True) -> pd.DataFrame:
    """
    Reads a point cloud from a file and returns it as a pandas DataFrame.
    Parameters:
    filename (Path): The path to the point cloud file. Supported file extensions are '.txt' and '.las'.
    Returns:
    pd.DataFrame: A DataFrame containing the point cloud data with columns:
        - 'X': X coordinates
        - 'Y': Y coordinates
        - 'Z': Z coordinates
        - 'zenith': Zenith angle (in degrees, range: 0 to 180)
        - 'azimuth': Azimuth angle (in degrees, range: 0 to 360)
        - 'rangemeter': Range in meters
        - 'Intensity': Intensity of the return
        - 'Return Number': Return number
    Raises:
    ValueError: If the file extension is not supported.
    """
    if filename.suffix == '.txt' and dataset_name == 'MANGROVE':
        print("Reading a text file - for Mongrove roots.")
        # Try reading the file assuming there is a header
        df = pd.read_csv(filename, sep=',')
        column_names = ['X', 'Y', 'Z', 'zenith', 'azimuth', 'rangemeter', 'Intensity', 'Return Number']
        # Check if the first row looks like column names (e.g., by type or value checks)
        if set(df.columns).intersection(column_names): # check if any of the predefined column names are in the dataframe
            print("Header detected.")
        else:
            # print("No header detected. Now trying to read the file with predefined column names.")
            # df = pd.read_csv(filename, sep=',', names=column_names)
            print("No header detected. Now trying to add predefined column names to the dataframe.")
            df.columns = column_names
        
        if flip_mangrove: # LIDAR upside down, Flip the Z axis to match the orientation of the point cloud
            df['Z'] = -df['Z'] # flip the Z axis for mongrove datasets
            df['elevation'] = df['zenith'] - 90
            print("------Flipping of Z axis for mangrove dataset.-------------")
        else: # For the single scan of the mangrove forest, lidar was not upsidedown.
            print("------Not flipping of Z axis for mangrove dataset.-------------")
            df['elevation'] = 90 - df['zenith']
            
    elif filename.suffix == '.las' and dataset_name == 'HARVARD_FOREST':
        print("Reading a .las file, for Harvard Forest data.")
        with laspy.open(filename) as las_file:
            las = las_file.read()
            scale_factors = las.header.scales
            shifts = las.header.offsets
            las_x = las.X * scale_factors[0] + shifts[0]
            las_y = las.Y * scale_factors[1] + shifts[1]
            las_z = las.Z * scale_factors[2] + shifts[2]
            # If las.intensity are all zeros (InLUT3D data), assign intensity values by mean of the R/G/B channels
            las_intensity = las.intensity if any(las.intensity) else (las.red + las.green + las.blue) / 3
            
            data = {
            'X': las_x,
            'Y': las_y,
            'Z': las_z,
            'Intensity': las_intensity,
            'Return Number': np.array(las.return_number),}
            df = pd.DataFrame(data)
            df = add_angle_range_to_df(df)
            # If the point cloud contains r/g/b channels, assign them to the DataFrame
            if any(las.red) and any(las.green) and any(las.blue):
                df['r'] = las.red / max(las.red)
                df['g'] = las.green / max(las.green)
                df['b'] = las.blue / max(las.blue)

    elif filename.suffix == '.bin' and dataset_name == 'SEMANTICKITTI':
        print("Reading a .bin file, for SemanticKitti data.")
        points = np.fromfile(filename, dtype=np.float32)
        points = points.reshape((-1, 4))

        df = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity'])
        df = add_angle_range_to_df(df)

    elif filename.suffix == '.pts' and dataset_name == 'INLUT3D':
        print("Reading a .pts file, for In_LUT3D data.")
        df = read_pts_file(filename) # columns: ['x', 'y', 'z', 'r', 'g', 'b', 'class_id', 'instance_id']
        # prepare intensity(psedo), azimuth, zenith, elevation angles, rangemeter, and return number (psedo)
        df['Intensity'] = df[['r', 'g', 'b']].mean(axis=1) / 255.0 # normalize to [0, 1]
        df = add_angle_range_to_df(df)

    elif filename.suffix == '.txt' and dataset_name == 'SEMANTIC3D':
        print("Reading a .txt file, for Semantic3D data.")
        # Try reading the file assuming there is a header
        column_names = ['X', 'Y', 'Z', 'Intensity', 'r', 'g', 'b']
        df = pd.read_csv(filename, sep='\s+', names=column_names)
        print(f"Read {len(df)} points from {filename}")
        print(df.head())
        df = add_angle_range_to_df(df)
        

    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}; supported extensions are .txt, .las, .bin, and .pts")
    
    print(f"Read {len(df)} points from {filename}")
    print("---------Before preprocessing:----------")
    print(f"Number of points: {len(df)}")
    for col in df.columns:
        print(f"Range of {col}: {df[col].min():.3f} to {df[col].max():.3f}")

    
    return df
        


def clean_pcd_df_based_on_ir(df: pd.DataFrame, cut_percent: float=0.006) -> pd.DataFrame:
    """
    Clean the point cloud data based on intensity and rangemeter.
    
    Parameters:
    df (pd.DataFrame): The point cloud data DataFrame.
    cut_percent (float): The percentage of points to keep based on intensity and rangemeter.
    
    Returns:
    pd.DataFrame: The cleaned point cloud data DataFrame.
    """
    print(f"❗ Cleaning point cloud data based on {cut_percent * 100}% cut-off.")
    col_names = ['rangemeter', 'Intensity']
    bottom_values = [0.1, 0.001]
    # Calculate the cut-off values for each column
    cut_off_values = {}
    for col, bottom in zip(col_names, bottom_values):
        top = np.percentile(df[col], 100 - cut_percent * 100)
        cut_off_values[col] = (bottom, top)
    
    # Filter the DataFrame based on the cut-off values
    df_cleaned = df.copy()
    for col, (bottom, top) in cut_off_values.items():
        df_cleaned = df_cleaned[(df_cleaned[col] >= bottom) & (df_cleaned[col] <= top)]
        
    print(f"Filtered {col}: {len(df) - len(df_cleaned)} points removed based on intensity and range limits:")
    print(cut_off_values)
    print(f"👉 After cleaning, {len(df_cleaned)} / {len(df)} points remain.")
    return df_cleaned


def read_and_clean_pcd(filename: Path, 
                            cut_percent: float = 0.005,
                           clean_pc: bool = False,
                           dataset_name: str = 'MANGROVE',
                           flip_mangrove: bool = True) -> pd.DataFrame:
    """
    Preprocess a point cloud data file and map scalar field values to pixel coordinates.
    
    Args:
        filename (Path): The path to the point cloud file. Supported formats are .txt, .las, and .bin.
        cut_percent (float): The percentage of points to keep based on intensity and rangemeter.
        clean_pc (bool): Whether to clean the point cloud data based on intensity and rangemeter.
        flip_mangrove (bool): Whether to flip the Z axis for mangrove datasets.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the filtered and processed point cloud data with additional columns for pixel coordinates.
    
    The input CSV file is expected to have the following columns:
    - 'X': X coordinate of the point.
    - 'Y': Y coordinate of the point.
    - 'Z': Z coordinate of the point.
    - 'zenith': Zenith angle of the point.
    - 'azimuth': Azimuth angle of the point.
    - 'rangemeter': Range to the point in meters.
    - 'Intensity': Intensity value of the point.
    - 'Return Number': Return number of the point.
    
    The function performs the following steps:
    1. Reads the file into a pandas DataFrame.
    2. Filters the DataFrame to include only points with a return number of 1 and intensity between 50 and 1000.
    3. Maps the azimuth and zenith angles to pixel coordinates (x_pix, y_pix) within the bounds of a predefined canvas size.
    4. Ensures the pixel indices are within the bounds of the canvas.
    """
    
    # # Step 1: Read the file as a pandas DataFrame
    df = read_raw_point_cloud(filename, dataset_name, flip_mangrove)

    # Step 2: Clean the dataset
    if clean_pc:
        df_filtered = clean_pcd_df_based_on_ir(df, cut_percent=cut_percent)
    else:
        df_filtered = df.copy()

    # Normalize the intensity values: first, convert to float32, then normalize to (0.1, 1.0)
    intensity = df_filtered['Intensity'].to_numpy().astype('float64')
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    df_filtered['Intensity'] = intensity
    print(f"!!Intensity normalized to range: {intensity.min()} to {intensity.max()}!!")
    
    df_filtered_na_free = df_filtered.dropna()
    print(f"{len(df_filtered) - len(df_filtered_na_free)} / {len(df_filtered)} points have NaN values, filtered out.")

    print("---------After preprocessing:----------")
    print(f"Number of points: {len(df_filtered_na_free)}")
    for col in df_filtered_na_free.columns:
        print(f"Range of {col}: {df_filtered_na_free[col].min()} to {df_filtered_na_free[col].max()}")
    return df_filtered_na_free

# Example usage
if __name__ == "__main__":
    filename = Path(r"/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/harvard_forest_33/33_01/33_01.las")
    df = read_and_clean_pcd(filename)
    print(df.head())