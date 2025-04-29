import pandas as pd
from pathlib import Path
import laspy
import numpy as np

HORIZONTAL_FOV = 360.0
# ======For TLS data======
VERTICAL_FOV = 135.0
VERTICAL_ANGLE_RESOLUTION = 0.25
HORIZONTAL_ANGLE_RESOLUTION = 0.25

# ===For SemanticKitti data (Velodyne-HDL-64)===
# VERTICAL_FOV = 26.8
# VERTICAL_ANGLE_RESOLUTION = 0.4188
# HORIZONTAL_ANGLE_RESOLUTION = 0.08


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
    import numpy as np

    # Stack the input vectors into a single array
    points = np.column_stack((las_x, las_y, las_z))

    # Calculate the magnitudes of the vectors
    magnitudes = np.linalg.norm(points, axis=1)

    # Avoid division by zero by assigning 0 to zenith_angles where magnitudes are 0
    with np.errstate(divide='ignore', invalid='ignore'):
        zenith_angles = np.arccos(np.divide(points[:, 2], magnitudes, where=magnitudes != 0)) * 180 / np.pi
        zenith_angles[magnitudes == 0] = 0

    return zenith_angles # range from 0 to 180



def map_angle_to_pixel(azimuth: np.ndarray, elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map azimuth and elevation angles to pixel coordinates on the unwrapped image.
    
    Parameters:
        azimuth (np.ndarray): Azimuth angles in degrees.
        elevation (np.ndarray): Elevation angles in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y pixel coordinates

    """

    CANVAS_WIDTH = int(HORIZONTAL_FOV / HORIZONTAL_ANGLE_RESOLUTION) # 1440 for TLS data (360 azimuth range)
    CANVAS_HEIGHT = int(VERTICAL_FOV / VERTICAL_ANGLE_RESOLUTION) # 540 for TLS data (135 elevation range)

    # Map azimuth (e.g., 0-360 degrees) to x-coordinate (0 to CANVAS_WIDTH-1)
    x_pix = (azimuth / HORIZONTAL_ANGLE_RESOLUTION).astype(int)

    # Map elevation angle (e.g., -90 ~ 45 degrees) to y-coordinate (0 to CANVAS_HEIGHT-1), 
    # flipping the y-axis since elevation increases from bottom to top while 
    # pixel indices increase from top to bottom.
    if elevation.min() < -45: # Mangrove root
        y_pix = CANVAS_HEIGHT - ((elevation + 90) / VERTICAL_ANGLE_RESOLUTION).astype(int)
    else: # Harvard Forest or an upward facing dataset in mangrove root
        y_pix = CANVAS_HEIGHT - ((elevation + 45) / VERTICAL_ANGLE_RESOLUTION).astype(int)

    # Ensure pixel indices are within bounds, in case x_pix or y_pix goes beyond 540 or 1440
    x_pix = x_pix.clip(0, CANVAS_WIDTH - 1)
    y_pix = y_pix.clip(0, CANVAS_HEIGHT - 1)

    return x_pix, y_pix


def read_raw_point_cloud(filename: Path, flip_mangrove:bool=True) -> pd.DataFrame:
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
        - 'range1metres': Range in meters
        - 'Intensity': Intensity of the return
        - 'Return Number': Return number
    Raises:
    ValueError: If the file extension is not supported.
    """
    if filename.suffix == '.txt':
        print("Reading a text file - for Mongrove roots.")
        try:
            # Try reading the file assuming there is a header
            df = pd.read_csv(filename, sep=',')
            column_names = ['X', 'Y', 'Z', 'zenith', 'azimuth', 'range1metres', 'Intensity', 'Return Number']
            # Check if the first row looks like column names (e.g., by type or value checks)
            if set(df.columns).intersection(column_names): # check if any of the predefined column names are in the dataframe
                print("Header detected.")
            else:
                print("No header detected. Now trying to read the file with predefined column names.")
                df = pd.read_csv(filename, sep=',', names=column_names)
            
            if flip_mangrove: # LIDAR upside down, Flip the Z axis to match the orientation of the point cloud
                df['Z'] = -df['Z'] # flip the Z axis for mongrove datasets
                df['elevation'] = df['zenith'] - 90
            else: # For the single scan of the mangrove forest, lidar was not upsidedown.
                print("------Not flipping of Z axis for mangrove dataset.-------------")
                df['elevation'] = 90 - df['zenith']
        except Exception as e:
            print(f"Error reading file: {e}")
            
    elif filename.suffix == '.las':
        print("Reading a .las file, for Harvard Forest data.")
        with laspy.open(filename) as las_file:
            las = las_file.read()
            scale_factors = las.header.scales
            las_x = las.X * scale_factors[0]
            las_y = las.Y * scale_factors[1]
            las_z = las.Z * scale_factors[2]
            data = {
            'X': las_x,
            'Y': las_y,
            'Z': las_z,
            'Intensity': las.intensity,
            'Return Number': np.array(las.return_number),}
            df = pd.DataFrame(data)
            # Calculate azimuth in degrees
            df['azimuth'] = np.arctan2(las_y, las_x) * 180 / np.pi
            # Remap azimuth values: [0, 180] stays the same, [-1, -180] becomes [181, 360]
            df['azimuth'] = np.where(df['azimuth'] < 0, 360 + df['azimuth'], df['azimuth'])
            # Calculate zenith in degrees range: theoretically (0 to 180), pratically (0, 135)
            zenith_angles = calculate_zenith_angles(las_x, las_y, las_z)
            df['elevation'] = 90 - zenith_angles # range from -90 to 90, pratically (-45, 90)
            df['zenith'] = zenith_angles
            df['range1metres'] = (las_x ** 2 + las_y ** 2 + las_z ** 2) ** 0.5
    elif filename.suffix == '.bin':
        print("Reading a .bin file, for SemanticKitti data.")
        points = np.fromfile(filename, dtype=np.float32)
        points = points.reshape((-1, 4))

        df = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity'])
        pc_x = df['X'].values
        pc_y = df['Y'].values
        pc_z = df['Z'].values
        df['Return Number'] = 1
        df['azimuth'] = np.arctan2(pc_y, pc_x) * 180 / np.pi
        df['azimuth'] = np.where(df['azimuth'] < 0, 360 + df['azimuth'], df['azimuth'])
        zenith_angles = calculate_zenith_angles(pc_x, pc_y, pc_z)
        df['elevation'] = 90 - zenith_angles # range from -90 to 90, pratically (-45, 90)
        df['zenith'] = zenith_angles
        df['range1metres'] = (pc_x ** 2 + pc_y ** 2 + pc_z ** 2) ** 0.5
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")
    
    print(f"Read {len(df)} points from {filename}")
    print("---------Before preprocessing:----------")
    for col in df.columns:
        print(f"Range of {col}: {df[col].min()} to {df[col].max()}")

    
    return df
        


def preprocess_point_cloud(filename: Path, 
                           range1metres_min: float = 0.1, 
                           range1metres_max: float = 50.0,
                           clean_pc: bool = False,
                           flip_mangrove: bool = True) -> pd.DataFrame:
    """
    Preprocess a point cloud data file and map scalar field values to pixel coordinates.
    
    Parameters:
    filename (Path): The path to the point cloud data file in CSV format.
    range1metres_min (float): Minimum threshold for range1metres filtering, in meters.
    range1metres_max (float): Maximum threshold for range1metres filtering, in meters.
    clean_pc (bool): Whether to filter the point cloud data based on range1metres_min and range1metres_max.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the filtered and processed point cloud data with additional columns for pixel coordinates.
    
    The input CSV file is expected to have the following columns:
    - 'X': X coordinate of the point.
    - 'Y': Y coordinate of the point.
    - 'Z': Z coordinate of the point.
    - 'zenith': Zenith angle of the point.
    - 'azimuth': Azimuth angle of the point.
    - 'range1metres': Range to the point in meters.
    - 'Intensity': Intensity value of the point.
    - 'Return Number': Return number of the point.
    
    The function performs the following steps:
    1. Reads the file into a pandas DataFrame.
    2. Filters the DataFrame to include only points with a return number of 1 and intensity between 50 and 1000.
    3. Maps the azimuth and zenith angles to pixel coordinates (x_pix, y_pix) within the bounds of a predefined canvas size.
    4. Ensures the pixel indices are within the bounds of the canvas.
    """
    
    # # Step 1: Read the file as a pandas DataFrame
    df = read_raw_point_cloud(filename, flip_mangrove)

    # Step 2: Clean the dataset
    if clean_pc:
        df_filtered = df[
            (df['range1metres'] >= range1metres_min)
            & (df['range1metres'] <= range1metres_max)
        ]
        print("Filtered based on range1metres. (range1metres_min, range1metres_max):", range1metres_min, range1metres_max)
    else:
        df_filtered = df

    # Normalize the intensity values: first, convert to float32, then normalize to (0.1, 1.0)
    intensity = df_filtered['Intensity'].to_numpy().astype('float32')
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    df_filtered['Intensity'] = intensity
    print(f"!!Intensity normalized to range: {intensity.min()} to {intensity.max()}!!")
    
    print(f"Filtered out {len(df) - len(df_filtered)} / {len(df)} points based on range1metres.")
    print("---------After preprocessing:----------")
    for col in df_filtered.columns:
        print(f"Range of {col}: {df_filtered[col].min()} to {df_filtered[col].max()}")
    return df_filtered

# Example usage
if __name__ == "__main__":
    filename = Path(r"/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/harvard_forest_33/33_01/33_01.las")
    df = preprocess_point_cloud(filename)
    print(df.head())