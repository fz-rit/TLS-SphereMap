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


def inspect_camera_properties(df: pd.DataFrame):
    assert all(c in df.columns for c in ["x", "y", "z"])

    xyz = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1).astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = maxs - mins

    center_mean = np.mean(xyz, axis=0)
    center_med = np.median(xyz, axis=0)

    z_q = np.percentile(z, [1, 5, 25, 50, 75, 95, 99])
    z_span_robust = float(z_q[-1] - z_q[0])

    radial_xy = np.sqrt((x - center_med[0]) ** 2 + (y - center_med[1]) ** 2)
    r_q = np.percentile(radial_xy, [50, 75, 90, 95, 99])

    cov = np.cov(xyz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    ev_ratio = eigvals / np.sum(eigvals)
    major_xy = eigvecs[:2, 0]
    major_az_deg = float(np.degrees(np.arctan2(major_xy[1], major_xy[0])))

    print("\n--- CAMERA-RELEVANT GEOMETRY ---")
    print(f"Points: {len(df):,}")
    print(f"AABB min: [{mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}]")
    print(f"AABB max: [{maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f}]")
    print(f"AABB span: [{spans[0]:.3f}, {spans[1]:.3f}, {spans[2]:.3f}]")
    print(f"Center(mean): [{center_mean[0]:.3f}, {center_mean[1]:.3f}, {center_mean[2]:.3f}]")
    print(f"Center(median): [{center_med[0]:.3f}, {center_med[1]:.3f}, {center_med[2]:.3f}]")
    print(
        "Z quantiles (1,5,25,50,75,95,99): "
        + ", ".join(f"{v:.3f}" for v in z_q)
    )
    print("XY radial quantiles (50,75,90,95,99): " + ", ".join(f"{v:.3f}" for v in r_q))
    print(
        f"PCA variance ratio: [{ev_ratio[0]:.3f}, {ev_ratio[1]:.3f}, {ev_ratio[2]:.3f}]"
    )
    print(f"Major XY axis azimuth: {major_az_deg:.2f} deg")

    lookat_z = float(np.percentile(z, 45))
    radius_start = float(r_q[2] * 0.9)
    radius_end = float(r_q[3] * 2.2 + 0.25 * z_span_robust)
    elev_start = 20.0
    elev_end = 86.0
    az_start = major_az_deg - 70.0
    az_end = major_az_deg + 20.0

    print("\n--- RECOMMENDED DYNAMIC CAMERA SEED ---")
    print(f"lookat: [{center_med[0]:.4f}, {center_med[1]:.4f}, {lookat_z:.4f}]")
    print(f"radius_start: {radius_start:.4f}")
    print(f"radius_end:   {radius_end:.4f}")
    print(f"elev_start_deg: {elev_start:.2f}")
    print(f"elev_end_deg:   {elev_end:.2f}")
    print(f"azimuth_start_deg: {az_start:.2f}")
    print(f"azimuth_end_deg:   {az_end:.2f}")
    print("zoom_start: 0.95")
    print("zoom_end:   0.55")


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
    inspect_camera_properties(df)


if __name__ == "__main__":
    main()