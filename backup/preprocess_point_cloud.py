import pandas as pd
from pathlib import Path
import laspy
import numpy as np

AZIMUTH_NORM_SCALE = 360
ZENITH_NORM_SCALE = 135
# Since degree resolution=0.25: 360/0.25=1440
CANVAS_WIDTH = 1440 
CANVAS_HEIGHT = 540

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

def read_point_cloud(filename: Path) -> pd.DataFrame:
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
    if filename.suffix == '.txt':
        # Define column names if there's no header
        column_names = ['X', 'Y', 'Z', 'zenith', 'azimuth', 'rangemeter', 'Intensity', 'Return Number']

        # Try reading the file assuming there is a header
        try:
            df = pd.read_csv(filename, sep=',')
            # Check if the first row looks like column names (e.g., by type or value checks)
            if set(df.columns).intersection(column_names):  # Adjust this condition as needed for your data
                print("Header detected.")
            else:
                print("No header detected, re-reading with predefined column names.")
                df = pd.read_csv(filename, sep=',', names=column_names)
        except Exception as e:
            print(f"Error reading file: {e}")
            df = pd.read_csv(filename, sep=',', names=column_names)
    elif filename.suffix == '.las':
        # Read the LAS file with laspy and then convert to dataframe.
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
            df['rangemeter'] = (las_x ** 2 + las_y ** 2 + las_z ** 2) ** 0.5
    elif filename.suffix == '.bin':
        # Read the SemanticKitti .bin file and convert to dataframe.
        points = np.fromfile(filename, dtype=np.float32)
        points = points.reshape((-1, 4))
        df = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity'])
        # Calculate azimuth in degrees
        df['azimuth'] = np.arctan2(df['Y'], df['X']) * 180 / np.pi
        # Remap azimuth values: [0, 180] stays the same, [-1, -180] becomes [181, 360]
        df['azimuth'] = np.where(df['azimuth'] < 0, 360 + df['azimuth'], df['azimuth'])
        # Calculate zenith in degrees range: (0 to 180)
        zenith_angles = calculate_zenith_angles(df['X'].values, df['Y'].values, df['Z'].values)
        df['elevation'] = 90 - zenith_angles
        df['zenith'] = zenith_angles
        df['rangemeter'] = (df['X'] ** 2 + df['Y'] ** 2 + df['Z'] ** 2) ** 0.5
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")
    
    print(f"Read {len(df)} points from {filename}")
    print(f"Columns: {df.columns}")
    print(f"Range of azimuth: {df['azimuth'].min()} to {df['azimuth'].max()}")
    print(f"Range of zenith: {df['zenith'].min()} to {df['zenith'].max()}")
    print(f"Range of rangemeter: {df['rangemeter'].min()} to {df['rangemeter'].max()}")
    print(f"Range of Intensity: {df['Intensity'].min()} to {df['Intensity'].max()}")
    print(f"Range of Return Number: {df['Return Number'].min()} to {df['Return Number'].max()}")
    print(f"Range of X: {df['X'].min()} to {df['X'].max()}")
    print(f"Range of Y: {df['Y'].min()} to {df['Y'].max()}")
    print(f"Range of Z: {df['Z'].min()} to {df['Z'].max()}")

    return df
        


def preprocess_point_cloud(filename: Path, 
                           rangemeter_min: float = 0.25, 
                           rangemeter_max: float = 15.0,
                           clean_pc: bool = True,
                           upside_down: bool = False) -> pd.DataFrame:
    """
    Preprocess a point cloud data file and map scalar field values to pixel coordinates.
    
    Parameters:
    filename (Path): The path to the point cloud data file in CSV format.
    rangemeter_min (float): Minimum threshold for rangemeter filtering, in meters.
    rangemeter_max (float): Maximum threshold for rangemeter filtering, in meters.
    
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
    df = read_point_cloud(filename)

    # Step 2: Clean the dataset
    if clean_pc:
        df_filtered = df[
            (df['rangemeter'] >= rangemeter_min)
            & (df['rangemeter'] <= rangemeter_max) 
        ]
    else:
        df_filtered = df

    # Step 3: Extract scanning angles and map to pixel coordinates
    # Map azimuth (e.g., 0-360 degrees) to x-coordinate (0 to CANVAS_WIDTH-1)
    df_filtered['x_pix'] = ((df_filtered['azimuth'] / AZIMUTH_NORM_SCALE) * (CANVAS_WIDTH - 1)).astype(int)

    # Map zenith (e.g., 0-135 degrees) to y-coordinate (0 to CANVAS_HEIGHT-1), flipping the y-axis
    if not upside_down: # for Harvard Forest datasets, y_pix = 0 is at the top, zenith ranges from 0 to 135, where 0 is the top and 135 is the bottom.
        df_filtered['y_pix'] = ((df_filtered['zenith'] / ZENITH_NORM_SCALE) * (CANVAS_HEIGHT - 1)).astype(int)
    else: # for Mangrove datasets, y_pix = 0 is at the bottom, zenith ranges from 0 to 135, where 0 is the bottom and 135 is the top.
        df_filtered['y_pix'] = (((ZENITH_NORM_SCALE - df_filtered['zenith']) / ZENITH_NORM_SCALE) * (CANVAS_HEIGHT - 1)).astype(int)

    # Ensure pixel indices are within bounds
    df_filtered['x_pix'] = df_filtered['x_pix'].clip(0, CANVAS_WIDTH - 1)
    df_filtered['y_pix'] = df_filtered['y_pix'].clip(0, CANVAS_HEIGHT - 1)

    return df_filtered

# Example usage
if __name__ == "__main__":
    filename = Path(r"/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/harvard_forest_33/33_01/33_01.las")
    df = preprocess_point_cloud(filename)
    print(df.head())