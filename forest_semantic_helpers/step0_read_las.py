import laspy
import numpy as np
from pathlib import Path
import pandas as pd

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




def read_pcd_file(file_path: Path, verbose: bool = False) -> pd.DataFrame:
    """Read LAS/LAZ file and return point cloud DataFrame."""
    with laspy.open(file_path) as f:
        points = f.read().points

    df = pd.DataFrame({
        'X': np.asarray(points.x).flatten(),
        'Y': np.asarray(points.y).flatten(),
        'Z': np.asarray(points.z).flatten(),
        'Intensity': np.asarray(points.intensity).flatten(),
        'Classification': np.asarray(points.classification).flatten()
    })

    if verbose:
        observe_df(df)

    return df



if __name__ == "__main__":
    pcd_dir = Path("/home/fzhcis/Downloads/ForestSemantic")
    pcd_path = pcd_dir / "Plot_5.las"
    pcd_df = read_pcd_file(pcd_path, verbose=True)