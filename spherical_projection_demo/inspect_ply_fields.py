from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from plyfile import PlyData


def read_ply_vertex_df(ply_path: str) -> pd.DataFrame:
    plydata = PlyData.read(ply_path)

    print("\n--- PLY HEADER INFO ---")
    print("Format:", plydata.text)
    print("Elements:")
    for elem in plydata.elements:
        print(f"  - {elem.name} ({elem.count} entries)")
        for prop in elem.properties:
            print(f"      • {prop.name} ({prop.dtype})")
    print()

    vertex = plydata["vertex"]
    data = {name: vertex[name] for name in vertex.data.dtype.names}
    return pd.DataFrame(data)


def print_scalar_stats(df: pd.DataFrame):
    print("Scalar statistics:\n")
    for name in df.columns:
        vals = df[name].to_numpy()
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        n_nan = np.isnan(vals).sum()
        print(
            f"{name:20s} | min={vmin:10.4f}  max={vmax:10.4f}  "
            f"mean={mean:10.4f}  std={std:10.4f}  NaN={n_nan}"
        )


def main():
    cfg_path = "spherical_unwrap.yaml"
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    ply_path = str(cfg["input_ply"])
    print(f"\nConfig: {cfg_path}")
    print(f"Reading: {ply_path}\n")

    df = read_ply_vertex_df(ply_path)
    print(f"\nTotal points: {len(df):,}\n")

    print_scalar_stats(df)


if __name__ == "__main__":
    main()