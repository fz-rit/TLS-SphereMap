import numpy as np
import pandas as pd
from plyfile import PlyData


def read_ply_raw(ply_path: str) -> pd.DataFrame:
    """
    Read PLY using plyfile and return vertex table as pandas DataFrame.
    """
    plydata = PlyData.read(ply_path)

    print("\n--- PLY HEADER INFO ---")
    print("Format:", plydata.text)
    print("Elements:")
    for elem in plydata.elements:
        print(f"  - {elem.name} ({elem.count} entries)")
        for prop in elem.properties:
            print(f"      • {prop.name} ({prop.dtype})")
    print()

    # Extract vertex element
    vertex = plydata["vertex"]
    data = {name: vertex[name] for name in vertex.data.dtype.names}

    df = pd.DataFrame(data)
    return df


def analyze_column(name: str, series: pd.Series):
    values = series.to_numpy()

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    n_nan = np.isnan(values).sum()

    print(f"{name:20s} | min={vmin:10.4f}  max={vmax:10.4f}  "
          f"mean={mean:10.4f}  std={std:10.4f}  NaN={n_nan}")

    lname = name.lower()

    # Angle heuristics
    if any(k in lname for k in ["az", "zen", "elev", "theta", "phi"]):
        print("   ↳ Angle-like field detected")

        if vmax <= 2*np.pi + 0.01:
            print("     Likely stored in radians")
        elif vmax <= 360 + 5:
            print("     Likely stored in degrees")

        if -180 <= vmin <= 0 and vmax <= 180:
            print("     Looks like -180° to 180°")
        elif 0 <= vmin and vmax <= 360:
            print("     Looks like 0° to 360°")
        elif 0 <= vmin and vmax <= 180:
            print("     Looks like 0° to 180°")
        elif -90 <= vmin and vmax <= 90:
            print("     Looks like -90° to 90° (elevation style)")
        print()

    # Range heuristics
    if "range" in lname or "dist" in lname:
        print("   ↳ Range-like field detected")
        if vmax > 1000:
            print("     Likely millimeters")
        elif vmax > 100:
            print("     Likely meters (large scene)")
        else:
            print("     Likely meters (TLS typical)")
        print()


def main():
    ply_path = "C:\\mylab\\data\\through_lidar_eye_demo\\33_01_filtered_normaled_curvature_0.06_roughness_0.06_segmap.ply"

    print(f"\nReading: {ply_path}\n")

    df = read_ply_raw(ply_path)

    print(f"\nTotal points: {len(df):,}\n")
    print("Scalar statistics:\n")

    for col in df.columns:
        analyze_column(col, df[col])


if __name__ == "__main__":
    main()