from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from plyfile import PlyData


def read_ply_vertex_df(ply_path: str) -> pd.DataFrame:
    ply = PlyData.read(ply_path)
    assert "vertex" in ply, f"No 'vertex' element in {ply_path}"
    v = ply["vertex"]
    names = v.data.dtype.names
    return pd.DataFrame({name: v[name] for name in names})


def resolve_frames_to_save(frames_to_have: str, n_cols: int, step_cols: int) -> list[int]:
    assert isinstance(frames_to_have, str)
    assert n_cols > 0 and step_cols > 0

    max_k = int(np.ceil((n_cols - 1) / step_cols))
    total_steps = max_k + 1

    token = frames_to_have.strip().lower()
    if token == "max":
        return list(range(total_steps))

    assert token.isdigit(), "progress.frames_to_have must be 'max' or numeric string"
    n_requested = int(token)
    assert n_requested > 0, "progress.frames_to_have must be > 0"

    n_take = min(n_requested, total_steps)
    if n_take == 1:
        return [max_k]

    return [int(i * (total_steps - 1) / (n_take - 1)) for i in range(n_take)]


def derive_camera_seed(df: pd.DataFrame) -> dict:
    assert all(c in df.columns for c in ["x", "y", "z"])

    xyz = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1).astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    center_xy = np.median(xyz[:, :2], axis=0)
    lookat_z_start = float(np.percentile(z, 40))
    lookat_z_end = float(np.percentile(z, 60))

    radial_xy = np.sqrt((x - center_xy[0]) ** 2 + (y - center_xy[1]) ** 2)
    r90 = float(np.percentile(radial_xy, 90))
    r95 = float(np.percentile(radial_xy, 95))

    z01 = float(np.percentile(z, 1))
    z99 = float(np.percentile(z, 99))
    z_span = max(z99 - z01, 1e-6)

    cov = np.cov(xyz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec_xy = eigvecs[:2, np.argsort(eigvals)[-1]]
    major_az_deg = float(np.degrees(np.arctan2(major_vec_xy[1], major_vec_xy[0])))

    return {
        "lookat_x": float(center_xy[0]),
        "lookat_y": float(center_xy[1]),
        "lookat_z_start": lookat_z_start,
        "lookat_z_end": lookat_z_end,
        "radius_start": max(r90 * 0.85, 1.0),
        "radius_end": max(r95 * 2.2 + 0.25 * z_span, 2.0),
        "elev_start_deg": 20.0,
        "elev_end_deg": 86.0,
        "azimuth_start_deg": major_az_deg - 70.0,
        "azimuth_end_deg": major_az_deg + 20.0,
        "zoom_start": 0.95,
        "zoom_end": 0.55,
    }


def main():
    cfg_path = "spherical_unwrap.yaml"
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    ply_path = str(cfg["input_ply"])
    width = int(cfg["grid"]["width"])
    step_cols = int(cfg["progress"]["step_cols"])
    frames_to_have = str(cfg["progress"]["frames_to_have"])
    camera_csv = str(cfg["render3d"]["camera_csv"])

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] PLY: {ply_path}")
    print(f"[INFO] Camera CSV: {camera_csv}")

    df = read_ply_vertex_df(ply_path)
    seed = derive_camera_seed(df)

    expected_ks = resolve_frames_to_save(frames_to_have, n_cols=width, step_cols=step_cols)
    cam = pd.read_csv(camera_csv)

    required = [
        "frame_idx",
        "frame_k",
        "phase",
        "cam_x",
        "cam_y",
        "cam_z",
        "lookat_x",
        "lookat_y",
        "lookat_z",
        "front_x",
        "front_y",
        "front_z",
        "up_x",
        "up_y",
        "up_z",
        "zoom",
    ]
    missing = [c for c in required if c not in cam.columns]
    assert len(missing) == 0, f"camera CSV missing columns: {missing}"

    phases = cam["phase"].unique()
    assert all(p in ["focus", "reveal", "survey"] for p in phases), f"Unknown phases in CSV: {phases}"

    assert len(cam) == len(expected_ks), (
        f"Row count mismatch: csv={len(cam)}, expected={len(expected_ks)}"
    )
    assert cam["frame_k"].astype(int).tolist() == [int(k) for k in expected_ks], "frame_k sequence mismatch"

    for axis in ["x", "y", "z"]:
        norm = np.sqrt(cam[f"front_x"] ** 2 + cam[f"front_y"] ** 2 + cam[f"front_z"] ** 2)
    assert np.allclose(norm.to_numpy(), 1.0, atol=1e-3), "front vectors are not normalized"

    radius = np.sqrt(
        (cam["cam_x"] - cam["lookat_x"]) ** 2
        + (cam["cam_y"] - cam["lookat_y"]) ** 2
        + (cam["cam_z"] - cam["lookat_z"]) ** 2
    )

    assert radius.iloc[-1] > radius.iloc[0], "camera radius should increase over frames"
    assert cam["zoom"].iloc[-1] < cam["zoom"].iloc[0], "zoom should decrease over frames"
    assert cam["front_z"].iloc[-1] < cam["front_z"].iloc[0], "front_z should move toward top-down"

    lookat_x_err = abs(float(np.median(cam["lookat_x"])) - seed["lookat_x"])
    lookat_y_err = abs(float(np.median(cam["lookat_y"])) - seed["lookat_y"])
    assert lookat_x_err < 1.0 and lookat_y_err < 1.0, "lookat center deviates from geometry expectation"

    r0_rel = abs(float(radius.iloc[0]) - seed["radius_start"]) / max(seed["radius_start"], 1e-6)
    r1_rel = abs(float(radius.iloc[-1]) - seed["radius_end"]) / max(seed["radius_end"], 1e-6)
    assert r0_rel < 0.2 and r1_rel < 0.2, "radius endpoints deviate from expected seed"

    z0 = float(cam["lookat_z"].iloc[0])
    z1 = float(cam["lookat_z"].iloc[-1])
    assert seed["lookat_z_start"] - 1.0 <= z0 <= seed["lookat_z_start"] + 1.0
    assert seed["lookat_z_end"] - 1.0 <= z1 <= seed["lookat_z_end"] + 1.0

    print("[PASS] Camera CSV sanity checks passed. 🎉")
    print(f"[PASS] frames={len(cam)}, k_range=[{cam['frame_k'].min()}..{cam['frame_k'].max()}]")
    print(f"[PASS] phases={sorted(cam['phase'].unique())}")
    print(f"[PASS] radius: start={radius.iloc[0]:.3f}, end={radius.iloc[-1]:.3f}")
    print(f"[PASS] zoom: start={cam['zoom'].iloc[0]:.3f}, end={cam['zoom'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
