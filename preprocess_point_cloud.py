import pandas as pd
from pathlib import Path
import laspy
import numpy as np

AZIMUTH_NORM_SCALE = 360
ZENITH_NORM_SCALE = 135
# Since degree resolution=0.25: 360/0.25=1440
CANVAS_WIDTH = 1440 
CANVAS_HEIGHT = 540

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
        - 'range1metres': Range in meters
        - 'Intensity': Intensity of the return
        - 'Return Number': Return number
    Raises:
    ValueError: If the file extension is not supported.
    """
    if filename.suffix == '.txt':
        # Define column names if there's no header
        column_names = ['X', 'Y', 'Z', 'zenith', 'azimuth', 'range1metres', 'Intensity', 'Return Number']

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
            # Calculate zenith in degrees (0 to 180)
            df['zenith'] = np.arctan2((las_x ** 2 + las_y ** 2) ** 0.5, las_z) * 180 / np.pi
            df['range1metres'] = (las_x ** 2 + las_y ** 2 + las_z ** 2) ** 0.5
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")

    return df
        


def preprocess_point_cloud(filename: Path, 
                           range1metres_min: float = 0.25, 
                           range1metres_max: float = 15.0,
                           clean_pc: bool = True,
                           upside_down: bool = False) -> pd.DataFrame:
    """
    Preprocess a point cloud data file and map scalar field values to pixel coordinates.
    
    Parameters:
    filename (Path): The path to the point cloud data file in CSV format.
    range1metres_min (float): Minimum threshold for range1metres filtering, in meters.
    range1metres_max (float): Maximum threshold for range1metres filtering, in meters.
    
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
    df = read_point_cloud(filename)

    # Step 2: Clean the dataset
    if clean_pc:
        # df_filtered = df[
        #     (df['Return Number'] == 1)
        #     & (df['Intensity'] >= 50)
        #     & (df['Intensity'] <= 1000) # experimentally determined
        #     & (df['range1metres'] >= range1metres_min)
        #     & (df['range1metres'] <= range1metres_max) 
        # ]
        df_filtered = df[
            (df['range1metres'] >= range1metres_min)
            & (df['range1metres'] <= range1metres_max) 
        ]
    else:
        df_filtered = df

    # Step 3: Extract scanning angles and map to pixel coordinates
    # Map azimuth (0-360 degrees) to x-coordinate (0 to CANVAS_WIDTH-1)
    df_filtered['x_pix'] = ((df_filtered['azimuth'] / AZIMUTH_NORM_SCALE) * (CANVAS_WIDTH - 1)).astype(int)

    # Map zenith (0-135 degrees) to y-coordinate (0 to CANVAS_HEIGHT-1), flipping the y-axis
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
    filename = Path(r"G:\My Drive\projects_with_Jan\for_Fei\harvard_forest_2021\033\33_01.las")
    df = preprocess_point_cloud(filename)
    print(df.head())