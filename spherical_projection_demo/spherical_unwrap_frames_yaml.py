from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from plyfile import PlyData

# 3D rendering
import open3d as o3d


# -------------------------
# I/O
# -------------------------
def read_ply_vertex_to_df(ply_path: str) -> pd.DataFrame:
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise RuntimeError(f"No 'vertex' element found in PLY: {ply_path}")
    v = ply["vertex"]
    names = v.data.dtype.names
    return pd.DataFrame({name: v[name] for name in names})


# -------------------------
# Mapping angles -> pixels
# -------------------------
def infer_angle_range(series: pd.Series, pad: float = 0.0) -> tuple[float, float]:
    vmin = float(np.nanmin(series.to_numpy()))
    vmax = float(np.nanmax(series.to_numpy()))
    return vmin - pad, vmax + pad


def angles_to_pixels(
    df: pd.DataFrame,
    az_col: str,
    ze_col: str,
    width: int,
    height: int,
    az_min: float,
    az_max: float,
    ze_min: float,
    ze_max: float,
    angles_in_degrees: bool = True,
) -> pd.DataFrame:
    az = df[az_col].astype(np.float64).to_numpy()
    ze = df[ze_col].astype(np.float64).to_numpy()

    if not angles_in_degrees:
        az = np.degrees(az)
        ze = np.degrees(ze)

    span = az_max - az_min
    if span <= 0:
        raise ValueError("az_max must be > az_min")
    az = ((az - az_min) % span) + az_min

    ze_span = ze_max - ze_min
    if ze_span <= 0:
        raise ValueError("ze_max must be > ze_min")

    az_n = (az - az_min) / span
    ze_n = (ze - ze_min) / ze_span

    az_n = np.clip(az_n, 0.0, 0.999999)
    ze_n = np.clip(ze_n, 0.0, 0.999999)

    col = (az_n * width).astype(np.int32)
    row = (ze_n * height).astype(np.int32)

    out = df.copy()
    out["col"] = col
    out["row"] = row
    return out


# -------------------------
# 2D Projection
# -------------------------
def project_count(df_pix: pd.DataFrame, width: int, height: int) -> np.ndarray:
    g = df_pix.groupby(["row", "col"], sort=False).size().rename("val")
    img = np.zeros((height, width), dtype=np.float32)

    idx = g.index.to_numpy()
    rr = np.fromiter((t[0] for t in idx), dtype=np.int32, count=len(idx))
    cc = np.fromiter((t[1] for t in idx), dtype=np.int32, count=len(idx))
    img[rr, cc] = g.to_numpy(dtype=np.float32)
    return img


def project_scalar(
    df_pix: pd.DataFrame,
    width: int,
    height: int,
    scalar_field: str,
    agg: str = "mean",
) -> np.ndarray:
    if scalar_field not in df_pix.columns:
        raise RuntimeError(f"scalar_field='{scalar_field}' not found. Available: {list(df_pix.columns)}")

    s = df_pix.groupby(["row", "col"], sort=False)[scalar_field]
    if agg == "mean":
        g = s.mean()
    elif agg == "max":
        g = s.max()
    elif agg == "min":
        g = s.min()
    elif agg == "median":
        g = s.median()
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    img = np.full((height, width), np.nan, dtype=np.float32)
    idx = g.index.to_numpy()
    rr = np.fromiter((t[0] for t in idx), dtype=np.int32, count=len(idx))
    cc = np.fromiter((t[1] for t in idx), dtype=np.int32, count=len(idx))
    img[rr, cc] = g.to_numpy(dtype=np.float32)
    return img


def compute_vmax(img: np.ndarray, percentile: float) -> float:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 1.0
    # ignore zeros if possible (helps count maps)
    if np.any(vals > 0):
        vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, percentile))


def save_frame_2d(
    img: np.ndarray,
    out_png: str,
    title: str,
    cmap: str,
    dpi: int,
    vmin: float,
    vmax: float,
    progress_col: int | None = None,
    draw_progress_line: bool = False,
):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(img, origin="upper", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    if draw_progress_line and progress_col is not None:
        plt.axvline(progress_col, linewidth=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


# -------------------------
# 3D Rendering (paired with frames)
# -------------------------
def build_open3d_pcd_from_df(
    df_sub: pd.DataFrame,
    use_rgb_if_available: bool,
    fallback_rgb=(0.9, 0.9, 0.9),
    max_points: int | None = None,
) -> o3d.geometry.PointCloud:
    if max_points is not None and len(df_sub) > max_points:
        df_sub = df_sub.sample(n=max_points, random_state=0)

    pts = np.stack([df_sub["x"].to_numpy(), df_sub["y"].to_numpy(), df_sub["z"].to_numpy()], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # colors if present
    if use_rgb_if_available and all(c in df_sub.columns for c in ["r", "g", "b"]):
        rgb = np.stack([df_sub["r"].to_numpy(), df_sub["g"].to_numpy(), df_sub["b"].to_numpy()], axis=1).astype(np.float64)
        # assume 0..255
        if rgb.max() > 1.0:
            rgb /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        col = np.array(fallback_rgb, dtype=np.float64).reshape(1, 3)
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(col, repeats=len(df_sub), axis=0))

    return pcd


def render_pcd_screenshot(
    pcd: o3d.geometry.PointCloud,
    out_png: str,
    width: int,
    height: int,
    point_size: float,
    bg_rgb=(0, 0, 0),
    camera=None,
):
    """
    Uses Open3D Visualizer in headless-ish mode (creates a hidden window).
    Works on most desktop Ubuntu setups. On true headless servers, this may fail
    unless EGL/OSMesa is configured.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg_rgb, dtype=np.float64)

    ctr = vis.get_view_control()
    if camera is not None:
        # Open3D-style camera parameters
        lookat = camera.get("lookat", [0, 0, 0])
        front = camera.get("front", [0, -1, 0])
        up = camera.get("up", [0, 0, 1])
        zoom = camera.get("zoom", 0.7)

        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(out_png, do_render=True)
    vis.destroy_window()


def smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def infer_scene_center_radius(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    pts = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1).astype(np.float64)
    center = np.mean(pts, axis=0)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    radius = 0.5 * float(np.linalg.norm(maxs - mins))
    radius = max(radius, 1e-6)
    return center, radius


def make_dynamic_camera(progress: float, center: np.ndarray, scene_radius: float, dyn_cfg: dict) -> dict:
    assert scene_radius > 0.0

    easing = str(dyn_cfg.get("easing", "smoothstep")).lower()
    s = smoothstep01(progress) if easing == "smoothstep" else float(np.clip(progress, 0.0, 1.0))

    radius_start_mult = float(dyn_cfg.get("radius_start_mult", 0.55))
    radius_end_mult = float(dyn_cfg.get("radius_end_mult", 2.4))
    elev_start_deg = float(dyn_cfg.get("elev_start_deg", 25.0))
    elev_end_deg = float(dyn_cfg.get("elev_end_deg", 86.0))
    azimuth_start_deg = float(dyn_cfg.get("azimuth_start_deg", -120.0))
    azimuth_end_deg = float(dyn_cfg.get("azimuth_end_deg", -30.0))
    zoom_start = float(dyn_cfg.get("zoom_start", 0.95))
    zoom_end = float(dyn_cfg.get("zoom_end", 0.55))

    radius = scene_radius * ((1.0 - s) * radius_start_mult + s * radius_end_mult)
    elev = np.radians((1.0 - s) * elev_start_deg + s * elev_end_deg)
    azim = np.radians((1.0 - s) * azimuth_start_deg + s * azimuth_end_deg)

    direction = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=np.float64,
    )
    cam_pos = center + radius * direction
    front = center - cam_pos
    front = front / max(np.linalg.norm(front), 1e-12)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(front, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    zoom = (1.0 - s) * zoom_start + s * zoom_end
    return {
        "lookat": center.tolist(),
        "front": front.tolist(),
        "up": up.tolist(),
        "zoom": float(zoom),
    }


def resolve_frames_to_save(frames_to_have: str, n_cols: int, step_cols: int) -> list[int]:
    assert isinstance(frames_to_have, str)
    assert n_cols > 0 and step_cols > 0

    max_k = int(np.ceil((n_cols - 1) / step_cols))
    total_steps = max_k + 1

    token = frames_to_have.strip().lower()
    if token == "max":
        return list(range(total_steps))

    assert token.isdigit(), "progress.frames_to_have must be 'max' or a numeric string like '10'"
    n_requested = int(token)
    assert n_requested > 0, "progress.frames_to_have numeric value must be > 0"

    n_take = min(n_requested, total_steps)
    if n_take == 1:
        return [max_k]

    picks = [int(i * (total_steps - 1) / (n_take - 1)) for i in range(n_take)]
    return picks


# -------------------------
# Main
# -------------------------
def main(cfg_path: str = "spherical_unwrap.yaml"):
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    ply_path = cfg["input_ply"]
    out_dir = cfg["out_dir"]

    az_field = cfg["fields"]["azimuth"]
    ze_field = cfg["fields"]["zenith"]
    angles_in_degrees = bool(cfg["fields"].get("angles_in_degrees", True))

    width = int(cfg["grid"]["width"])
    height = int(cfg["grid"]["height"])
    assert width > 0 and height > 0

    az_min = float(cfg["ranges"]["az_min"])
    az_max = float(cfg["ranges"]["az_max"])
    ze_min = cfg["ranges"]["ze_min"]
    ze_max = cfg["ranges"]["ze_max"]

    step_cols = int(cfg["progress"]["step_cols"])
    assert step_cols > 0
    frames_to_have = str(cfg["progress"]["frames_to_have"])

    proj_cfg = cfg.get("projection", {})
    proj_mode = proj_cfg.get("mode", "count")  # "count" or "scalar"
    scalar_field = proj_cfg.get("scalar_field", "range1metres")
    agg = proj_cfg.get("agg", "mean")

    render_cfg = cfg.get("render", {})
    cmap = render_cfg.get("cmap", "gray")
    dpi = int(render_cfg.get("dpi", 180))
    vmax_percentile = float(render_cfg.get("vmax_percentile", 99.5))
    draw_progress_line = bool(render_cfg.get("draw_progress_line", True))

    render3d = cfg.get("render3d", {})
    render3d_enabled = bool(render3d.get("enabled", False))
    render3d_subdir = render3d.get("out_subdir", "frames_3d")
    render3d_w = int(render3d.get("width", 1280))
    render3d_h = int(render3d.get("height", 720))
    assert render3d_w > 0 and render3d_h > 0
    render3d_point_size = float(render3d.get("point_size", 2.0))
    bg_rgb = tuple(render3d.get("background_rgb", [0, 0, 0]))
    use_rgb_if_available = bool(render3d.get("use_rgb_if_available", True))
    fallback_rgb = tuple(render3d.get("fallback_rgb", [0.9, 0.9, 0.9]))
    max_points = render3d.get("max_points", 200000)
    if max_points is None:
        max_points = None
    else:
        max_points = int(max_points)

    dynamic_camera_cfg = render3d.get("dynamic_camera", {})
    dynamic_camera_enabled = bool(dynamic_camera_cfg.get("enabled", True))
    camera = render3d.get("camera", None)

    print(f"[INFO] Reading PLY: {ply_path}")
    df = read_ply_vertex_to_df(ply_path)
    print(f"[INFO] Points: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")

    scene_center, scene_radius = infer_scene_center_radius(df)

    if az_field not in df.columns or ze_field not in df.columns:
        raise RuntimeError(
            f"Angle fields missing. Need az='{az_field}', ze='{ze_field}'. Available: {list(df.columns)}"
        )

    if isinstance(ze_min, str) and ze_min.lower() == "auto":
        ze_min, _ = infer_angle_range(df[ze_field])
        print(f"[INFO] ze_min=auto -> {ze_min:.6f}")
    else:
        ze_min = float(ze_min)

    if isinstance(ze_max, str) and ze_max.lower() == "auto":
        _, ze_max = infer_angle_range(df[ze_field])
        print(f"[INFO] ze_max=auto -> {ze_max:.6f}")
    else:
        ze_max = float(ze_max)

    dfp = angles_to_pixels(
        df,
        az_col=az_field,
        ze_col=ze_field,
        width=width,
        height=height,
        az_min=az_min,
        az_max=az_max,
        ze_min=ze_min,
        ze_max=ze_max,
        angles_in_degrees=angles_in_degrees,
    )

    # stable contrast from full projection
    if proj_mode == "count":
        full_img = project_count(dfp, width, height)
        vmin = 0.0
        vmax = compute_vmax(full_img, vmax_percentile)
        if vmax <= 0:
            vmax = 1.0
    elif proj_mode == "scalar":
        full_img = project_scalar(dfp, width, height, scalar_field=scalar_field, agg=agg)
        finite = full_img[np.isfinite(full_img)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = compute_vmax(full_img, vmax_percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        raise ValueError("projection.mode must be 'count' or 'scalar'")

    print(f"[INFO] Projection mode: {proj_mode}")
    print(f"[INFO] Contrast: vmin={vmin:.4f}, vmax(p{vmax_percentile})={vmax:.4f}")
    if render3d_enabled:
        print("[INFO] 3D screenshots: enabled")

    # output dirs
    out_dir_2d = os.path.join(out_dir, "frames_2d")
    out_dir_3d = os.path.join(out_dir, render3d_subdir)

    n_cols = width
    frames_to_save = resolve_frames_to_save(frames_to_have, n_cols=n_cols, step_cols=step_cols)

    assert len(frames_to_save) > 0
    n_frames = len(frames_to_save)

    for frame_idx, k in enumerate(frames_to_save):
        max_col = min(n_cols - 1, k * step_cols)
        sub = dfp[dfp["col"] <= max_col]

        # ----- 2D frame
        if proj_mode == "count":
            img2d = project_count(sub, width, height)
        else:
            img2d = project_scalar(sub, width, height, scalar_field=scalar_field, agg=agg)

        out_png_2d = os.path.join(out_dir_2d, f"frame_k{k:04d}_col{max_col:04d}.png")
        title = f"Unwrap buildup: az cols 0..{max_col}, ze[{ze_min:.1f},{ze_max:.1f}]"
        save_frame_2d(
            img2d,
            out_png_2d,
            title=title,
            cmap=cmap,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            progress_col=max_col,
            draw_progress_line=draw_progress_line,
        )
        print(f"[WROTE] 2D {out_png_2d}")

        # ----- 3D screenshot (same subset)
        if render3d_enabled:
            if dynamic_camera_enabled:
                t = frame_idx / max(n_frames - 1, 1)
                frame_camera = make_dynamic_camera(t, scene_center, scene_radius, dynamic_camera_cfg)
            else:
                frame_camera = camera

            pcd = build_open3d_pcd_from_df(
                sub,
                use_rgb_if_available=use_rgb_if_available,
                fallback_rgb=fallback_rgb,
                max_points=max_points,
            )
            out_png_3d = os.path.join(out_dir_3d, f"frame_k{k:04d}_col{max_col:04d}.png")
            render_pcd_screenshot(
                pcd,
                out_png=out_png_3d,
                width=render3d_w,
                height=render3d_h,
                point_size=render3d_point_size,
                bg_rgb=bg_rgb,
                camera=frame_camera,
            )
            print(f"[WROTE] 3D {out_png_3d}")

    print(f"[DONE] Saved 2D frames to: {out_dir_2d}")
    if render3d_enabled:
        print(f"[DONE] Saved 3D frames to: {out_dir_3d}")


if __name__ == "__main__":
    main("spherical_unwrap.yaml")