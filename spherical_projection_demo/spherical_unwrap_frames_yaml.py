from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from PIL import Image
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
    out_png: str | None,
    title: str,
    cmap: str,
    dpi: int,
    vmin: float,
    vmax: float,
    progress_col: int | None = None,
    draw_progress_line: bool = False,
    opposite_col: int | None = None,
    az_min: float = 0.0,
    az_max: float = 360.0,
    ze_min: float = 0.0,
    ze_max: float = 180.0,
    return_pil: bool = False,
) -> Image.Image | None:
    """Save 2D frame and/or return as PIL Image.
    
    Args:
        out_png: Path to save PNG file, or None to skip saving
        return_pil: If True, return PIL Image object
    
    Returns:
        PIL Image if return_pil=True, else None
    """
    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(img, origin="upper", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                   extent=[az_min, az_max, ze_max, ze_min])  # extent for axis labels
    
    if draw_progress_line and progress_col is not None:
        # Convert column index to azimuth angle
        progress_az = az_min + progress_col * (az_max - az_min) / img.shape[1]
        ax.axvline(progress_az, color="cyan", linewidth=1.5, alpha=0.8, label="Current scan angle")
        if opposite_col is not None:
            opposite_az = az_min + opposite_col * (az_max - az_min) / img.shape[1]
            ax.axvline(opposite_az, color="cyan", linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Azimuth (degrees)", fontsize=10)
    ax.set_ylabel("Zenith (degrees)", fontsize=10)
    ax.set_title(title, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value", fontsize=9)
    
    plt.tight_layout()
    
    pil_img = None
    if return_pil or out_png is not None:
        # Render to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        pil_img = Image.open(buf).copy()
        buf.close()
        
        if out_png is not None:
            pil_img.save(out_png)
    
    plt.close()
    return pil_img if return_pil else None


# -------------------------
# 3D Rendering (paired with frames)
# -------------------------
def build_open3d_pcd_from_2d_image(
    df_sub: pd.DataFrame,
    img2d: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    max_points: int | None = None,
    fallback_rgb=(0.5, 0.5, 0.5),
) -> o3d.geometry.PointCloud:
    """Build point cloud with colors mapped from 2D projection image."""
    if max_points is not None and len(df_sub) > max_points:
        df_sub = df_sub.sample(n=max_points, random_state=0)

    pts = np.stack([df_sub["x"].to_numpy(), df_sub["y"].to_numpy(), df_sub["z"].to_numpy()], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Get matplotlib colormap and apply to 2D image
    cm = plt.get_cmap(cmap)
    img_norm = np.clip((img2d - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    img_rgb = cm(img_norm)[:, :, :3]  # RGB only, drop alpha channel

    # Map colors from 2D image to points using row/col indices
    rows = df_sub["row"].to_numpy()
    cols = df_sub["col"].to_numpy()
    colors = np.zeros((len(df_sub), 3), dtype=np.float64)

    for i in range(len(df_sub)):
        r, c = int(rows[i]), int(cols[i])
        if 0 <= r < img2d.shape[0] and 0 <= c < img2d.shape[1] and np.isfinite(img2d[r, c]):
            colors[i] = img_rgb[r, c]
        else:
            colors[i] = fallback_rgb

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_pcd_screenshot(
    pcd: o3d.geometry.PointCloud,
    out_png: str | None,
    width: int,
    height: int,
    point_size: float,
    bg_rgb=(0, 0, 0),
    lookat=None,
    front=None,
    up=None,
    zoom=0.7,
    return_pil: bool = False,
) -> Image.Image | None:
    """Render 3D point cloud screenshot and/or return as PIL Image.
    
    Args:
        out_png: Path to save PNG file, or None to skip saving
        return_pil: If True, return PIL Image object
    
    Returns:
        PIL Image if return_pil=True, else None
    """
    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg_rgb, dtype=np.float64)

    ctr = vis.get_view_control()
    if lookat is not None:
        ctr.set_lookat(lookat)
    if front is not None:
        ctr.set_front(front)
    if up is not None:
        ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    # Capture to temporary location if we need PIL image
    pil_img = None
    if return_pil or out_png is not None:
        import tempfile
        if out_png is None:
            # Need temporary file for capture
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
            os.close(tmp_fd)
            vis.capture_screen_image(tmp_path, do_render=True)
            pil_img = Image.open(tmp_path).copy()
            os.unlink(tmp_path)
        else:
            vis.capture_screen_image(out_png, do_render=True)
            if return_pil:
                pil_img = Image.open(out_png).copy()
    
    vis.destroy_window()
    return pil_img if return_pil else None


def compute_fixed_camera_params(df: pd.DataFrame) -> dict:
    """Compute center and reasonable camera positions from point cloud."""
    xyz = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1).astype(np.float64)
    center = np.median(xyz, axis=0)
    ptp = np.ptp(xyz, axis=0)
    max_span = float(np.max(ptp))
    
    return {
        "lookat": center.tolist(),
        "max_span": max_span,
    }


def create_gif_from_pil_images(
    images: list[Image.Image],
    out_gif: str,
    duration: int = 100,
    loop: int = 0,
    hold_last_frame: int = 0,
):
    """Create animated GIF from PIL images.
    
    Args:
        images: List of PIL Image objects
        out_gif: Output GIF path
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
        hold_last_frame: Number of additional times to repeat the last frame (default: 0)
    """
    if not images:
        print(f"[WARNING] No images to create GIF: {out_gif}")
        return
    
    # Duplicate the last frame to hold it longer at the end of the animation
    # This creates a pause effect before looping back to the beginning
    if hold_last_frame > 0:
        images = images + [images[-1]] * hold_last_frame
    
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    images[0].save(
        out_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )
    print(f"[WROTE GIF] {out_gif} ({len(images)} frames)")


def resolve_frames_to_save(
    frames_to_have: str, 
    n_cols: int, 
    step_degrees: float,
    angular_resolution: float,
    tls_dual_scan: bool,
) -> list[int]:
    assert isinstance(frames_to_have, str)
    assert n_cols > 0
    
    # With dual-scan, physical rotation is only 180° (half the columns)
    # Without dual-scan, full 360° rotation
    effective_cols = (n_cols // 2) if tls_dual_scan else n_cols
    
    # Convert step_degrees to step_cols
    step_cols = max(1, int(np.round(step_degrees / angular_resolution)))
    
    max_k = int(np.ceil((effective_cols - 1) / step_cols))
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

    az_min = float(cfg["ranges"]["az_min"])
    az_max = float(cfg["ranges"]["az_max"])
    ze_min_cfg = cfg["ranges"]["ze_min"]
    ze_max_cfg = cfg["ranges"]["ze_max"]
    
    # Angular resolution for adaptive grid
    angular_resolution = float(cfg["grid"].get("angular_resolution", 0.25))  # degrees per pixel
    
    print(f"[INFO] Reading PLY: {ply_path}")
    df = read_ply_vertex_to_df(ply_path)
    print(f"[INFO] Points: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")

    if az_field not in df.columns or ze_field not in df.columns:
        raise RuntimeError(
            f"Angle fields missing. Need az='{az_field}', ze='{ze_field}'. Available: {list(df.columns)}"
        )

    # Resolve zenith range (may be auto)
    if isinstance(ze_min_cfg, str) and ze_min_cfg.lower() == "auto":
        ze_min, _ = infer_angle_range(df[ze_field])
        print(f"[INFO] ze_min=auto -> {ze_min:.6f}")
    else:
        ze_min = float(ze_min_cfg)

    if isinstance(ze_max_cfg, str) and ze_max_cfg.lower() == "auto":
        _, ze_max = infer_angle_range(df[ze_field])
        print(f"[INFO] ze_max=auto -> {ze_max:.6f}")
    else:
        ze_max = float(ze_max_cfg)

    # Compute adaptive grid dimensions based on angular resolution
    az_span = az_max - az_min
    ze_span = ze_max - ze_min
    width = int(np.ceil(az_span / angular_resolution))
    height = int(np.ceil(ze_span / angular_resolution))
    
    print(f"[INFO] Angular resolution: {angular_resolution}°")
    print(f"[INFO] Azimuth range: [{az_min:.1f}, {az_max:.1f}]° -> {width} columns")
    print(f"[INFO] Zenith range: [{ze_min:.1f}, {ze_max:.1f}]° -> {height} rows")

    step_degrees = float(cfg["progress"].get("step_degrees", 0.5))  # degrees per frame step
    frames_to_have = str(cfg["progress"]["frames_to_have"])
    tls_dual_scan = bool(cfg["progress"].get("tls_dual_scan", True))  # TLS scans opposite azimuths simultaneously
    
    print(f"[INFO] Step size: {step_degrees}° per frame")
    print(f"[INFO] TLS dual-scan mode: {tls_dual_scan} (physical rotation: {'180°' if tls_dual_scan else '360°'})")

    proj_cfg = cfg.get("projection", {})
    proj_mode = proj_cfg.get("mode", "count")  # "count" or "scalar"
    scalar_field = proj_cfg.get("scalar_field", "range1metres")
    agg = proj_cfg.get("agg", "mean")

    render_cfg = cfg.get("render", {})
    cmap = render_cfg.get("cmap", "gray")
    dpi = int(render_cfg.get("dpi", 180))
    vmax_percentile = float(render_cfg.get("vmax_percentile", 99.5))
    draw_progress_line = bool(render_cfg.get("draw_progress_line", True))
    save_individual_frames = bool(render_cfg.get("save_individual_frames", False))
    gif_duration = int(render_cfg.get("gif_duration_ms", 100))  # milliseconds per frame
    gif_loop = int(render_cfg.get("gif_loop", 0))  # 0 = infinite loop

    render3d = cfg.get("render3d", {})
    render3d_enabled = bool(render3d.get("enabled", False))
    render3d_side_subdir = render3d.get("side_subdir", "frames_3d_side")
    render3d_top_subdir = render3d.get("top_subdir", "frames_3d_top")
    render3d_w = int(render3d.get("width", 1280))
    render3d_h = int(render3d.get("height", 720))
    assert render3d_w > 0 and render3d_h > 0
    render3d_point_size = float(render3d.get("point_size", 2.0))
    bg_rgb = tuple(render3d.get("background_rgb", [0, 0, 0]))
    max_points = render3d.get("max_points", 200000)
    if max_points is not None:
        max_points = int(max_points)

    assert width > 0 and height > 0

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
    print(f"[INFO] Save individual frames: {save_individual_frames}")
    print(f"[INFO] GIF animation: enabled (duration={gif_duration}ms/frame)")
    if render3d_enabled:
        print("[INFO] 3D screenshots: enabled")

    # output dirs
    out_dir_2d = os.path.join(out_dir, "frames_2d")
    out_dir_3d_side = os.path.join(out_dir, render3d_side_subdir)
    out_dir_3d_top = os.path.join(out_dir, render3d_top_subdir)

    n_cols = width
    frames_to_save = resolve_frames_to_save(
        frames_to_have=frames_to_have, 
        n_cols=n_cols, 
        step_degrees=step_degrees,
        angular_resolution=angular_resolution,
        tls_dual_scan=tls_dual_scan,
    )
    assert len(frames_to_save) > 0
    n_frames = len(frames_to_save)
    
    print(f"[INFO] Exporting {n_frames} frames")

    # Convert step_degrees to step_cols for frame iteration
    step_cols = max(1, int(np.round(step_degrees / angular_resolution)))
    
    # Storage for GIF frames
    frames_2d_pil = []
    frames_3d_side_pil = []
    frames_3d_top_pil = []

    # Compute fixed camera parameters from full point cloud
    if render3d_enabled:
        cam_params = compute_fixed_camera_params(df)
        lookat = cam_params["lookat"]
        max_span = cam_params["max_span"]
        
        # Side view: looking at x-z plane (front view)
        side_front = [0.0, -1.0, 0.0]  # look from -Y toward +Y
        side_up = [0.0, 0.0, 1.0]     # Z is up
        side_zoom = 0.7
        
        # Top view: looking down from above
        top_front = [0.0, 0.0, -1.0]  # look from +Z toward -Z
        top_up = [0.0, 1.0, 0.0]      # Y is up in screen space
        top_zoom = 0.7

    for frame_idx, k in enumerate(frames_to_save):
        max_col = min(n_cols - 1, k * step_cols)
        
        # TLS typically scans opposite azimuths simultaneously
        # (0° and 180°, 0.25° and 180.25°, etc.)
        if tls_dual_scan:
            # Include columns 0 to max_col AND their 180° opposites
            half_width = width // 2
            opposite_cols = set()
            for c in range(max_col + 1):
                opposite_c = (c + half_width) % width
                opposite_cols.add(opposite_c)
            
            # Filter points: include if col <= max_col OR col is an opposite
            sub = dfp[(dfp["col"] <= max_col) | (dfp["col"].isin(opposite_cols))]
        else:
            # Sequential scan: just columns 0 to max_col
            sub = dfp[dfp["col"] <= max_col]

        # ----- 2D frame
        if proj_mode == "count":
            img2d = project_count(sub, width, height)
        else:
            img2d = project_scalar(sub, width, height, scalar_field=scalar_field, agg=agg)

        out_png_2d = os.path.join(out_dir_2d, f"frame_k{k:04d}_col{max_col:04d}.png") if save_individual_frames else None
        if tls_dual_scan:
            # Calculate angular position (physical rotation is only 180°)
            current_angle = max_col * angular_resolution
            title = f"TLS Dual-Scan | Rotation: {current_angle:.1f}° | Coverage: 0-{current_angle:.1f}° & 180-{current_angle + 180:.1f}°"
            opposite_col = (max_col + width // 2) % width
        else:
            current_angle = max_col * angular_resolution
            title = f"Sequential Scan | Azimuth: 0-{current_angle:.1f}° | Zenith: {ze_min:.1f}-{ze_max:.1f}°"
            opposite_col = None
        
        pil_2d = save_frame_2d(
            img2d,
            out_png_2d,
            title=title,
            cmap=cmap,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            progress_col=max_col,
            draw_progress_line=draw_progress_line,
            opposite_col=opposite_col,
            az_min=az_min,
            az_max=az_max,
            ze_min=ze_min,
            ze_max=ze_max,
            return_pil=True,
        )
        frames_2d_pil.append(pil_2d)
        
        if save_individual_frames:
            print(f"[WROTE] 2D {out_png_2d}")

        # ----- 3D screenshots (side + top views)
        if render3d_enabled:
            pcd = build_open3d_pcd_from_2d_image(
                df_sub=sub,
                img2d=img2d,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                max_points=max_points,
                fallback_rgb=tuple(render3d.get("fallback_rgb", [0.5, 0.5, 0.5])),
            )
            
            # Side view
            out_png_side = os.path.join(out_dir_3d_side, f"frame_k{k:04d}_col{max_col:04d}.png") if save_individual_frames else None
            pil_side = render_pcd_screenshot(
                pcd,
                out_png=out_png_side,
                width=render3d_w,
                height=render3d_h,
                point_size=render3d_point_size,
                bg_rgb=bg_rgb,
                lookat=lookat,
                front=side_front,
                up=side_up,
                zoom=side_zoom,
                return_pil=True,
            )
            frames_3d_side_pil.append(pil_side)
            
            if save_individual_frames:
                print(f"[WROTE] 3D side {out_png_side}")
            
            # Top view
            out_png_top = os.path.join(out_dir_3d_top, f"frame_k{k:04d}_col{max_col:04d}.png") if save_individual_frames else None
            pil_top = render_pcd_screenshot(
                pcd,
                out_png=out_png_top,
                width=render3d_w,
                height=render3d_h,
                point_size=render3d_point_size,
                bg_rgb=bg_rgb,
                lookat=lookat,
                front=top_front,
                up=top_up,
                zoom=top_zoom,
                return_pil=True,
            )
            frames_3d_top_pil.append(pil_top)
            
            if save_individual_frames:
                print(f"[WROTE] 3D top {out_png_top}")

    # Generate GIF files
    gif_2d = os.path.join(out_dir, "animation_2d.gif")
    create_gif_from_pil_images(frames_2d_pil, gif_2d, duration=gif_duration, loop=gif_loop, hold_last_frame=9)
    
    if render3d_enabled:
        gif_3d_side = os.path.join(out_dir, "animation_3d_side.gif")
        gif_3d_top = os.path.join(out_dir, "animation_3d_top.gif")
        create_gif_from_pil_images(frames_3d_side_pil, gif_3d_side, duration=gif_duration, loop=gif_loop, hold_last_frame=9)
        create_gif_from_pil_images(frames_3d_top_pil, gif_3d_top, duration=gif_duration, loop=gif_loop, hold_last_frame=9)
    
    if save_individual_frames:
        print(f"[DONE] Saved 2D frames to: {out_dir_2d}")
        if render3d_enabled:
            print(f"[DONE] Saved 3D side views to: {out_dir_3d_side}")
            print(f"[DONE] Saved 3D top views to: {out_dir_3d_top}")
    
    print(f"[DONE] Animation saved to: {out_dir}")


if __name__ == "__main__":
    main("spherical_unwrap.yaml")