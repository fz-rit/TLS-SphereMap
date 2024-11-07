import pandas as pd
from pathlib import Path

AZIMUTH_NORM_SCALE = 360
ZENITH_NORM_SCALE = 135
# Since degree resolution=0.25: 360/0.25=1440
CANVAS_WIDTH = 1440 
CANVAS_HEIGHT = 540

def preprocess_point_cloud(filename: Path) -> pd.DataFrame:
    """
    Preprocess a point cloud data file and map scalar field values to pixel coordinates.
    Parameters:
    filename (Path): The path to the point cloud data file in CSV format.
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
    
    # Step 1: Read the file as a pandas DataFrame
    df = pd.read_csv(
        filename,
        delimiter=',', 
        names=['X', 'Y', 'Z', 'zenith', 'azimuth', 'range1metres', 'Intensity', 'Return Number']
    )

    # Step 2: Clean the dataset
    df_filtered = df[
        (df['Return Number'] == 1) &
        (df['Intensity'] >= 50) &
        (df['Intensity'] <= 1000) & 
        (df['range1metres'] >= 0.25) & 
        (df['range1metres'] <= 10) 
        
    ]

    # Step 3: Extract scanning angles and map to pixel coordinates
    # Map azimuth (0-360 degrees) to x-coordinate (0 to CANVAS_WIDTH-1)
    df_filtered['x_pix'] = ((df_filtered['azimuth'] / AZIMUTH_NORM_SCALE) * (CANVAS_WIDTH - 1)).astype(int)

    # Map zenith (0-135 degrees) to y-coordinate (0 to CANVAS_HEIGHT-1), flipping the y-axis
    df_filtered['y_pix'] = (((ZENITH_NORM_SCALE - df_filtered['zenith']) / ZENITH_NORM_SCALE) * (CANVAS_HEIGHT - 1)).astype(int)

    # Ensure pixel indices are within bounds
    df_filtered['x_pix'] = df_filtered['x_pix'].clip(0, CANVAS_WIDTH - 1)
    df_filtered['y_pix'] = df_filtered['y_pix'].clip(0, CANVAS_HEIGHT - 1)

    return df_filtered