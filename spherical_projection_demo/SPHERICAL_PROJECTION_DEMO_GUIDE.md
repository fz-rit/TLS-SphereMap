# Spherical Projection Demo Guide

## Folder structure

```text
spherical_projection_demo/
├─ inspect_ply_fields.py
├─ generate_dynamic_camera_csv.py
├─ sanity_check_camera_csv.py (optional standalone validator)
├─ spherical_unwrap_frames_yaml.py
├─ spherical_unwrap.yaml
└─ SPHERICAL_PROJECTION_DEMO_GUIDE.md
```

## Script roles

- `inspect_ply_fields.py`
  - Reads `input_ply` from `spherical_unwrap.yaml`.
  - Prints scalar stats and camera-relevant geometry diagnostics (AABB, robust center, radial quantiles, PCA orientation, recommended camera seed).

- `generate_dynamic_camera_csv.py`
  - Reads `spherical_unwrap.yaml`.
  - Resolves output frames from `progress.frames_to_have` and `progress.step_cols`.
  - Derives scene-aware dynamic camera poses.
  - Saves camera poses to `render3d.camera_csv` as CSV.
  - Validates generated camera CSV (trajectory trends, geometry alignment, vector normalization).

- `spherical_unwrap_frames_yaml.py`
  - Reads `spherical_unwrap.yaml`.
  - Loads camera poses from `render3d.camera_csv`.
  - Exports 2D unwrap frames and matched 3D screenshots.

- `sanity_check_camera_csv.py` (optional)
  - Standalone validation script.
  - Reads `input_ply`, `progress.*`, and `render3d.camera_csv` from `spherical_unwrap.yaml`.
  - Validates camera CSV shape, frame alignment, trajectory trends, and geometry-based expectations.
  - Note: Validation is now integrated into `generate_dynamic_camera_csv.py`.

## Required config keys (in `spherical_unwrap.yaml`)

- `progress.step_cols`
- `progress.frames_to_have` (`"max"` or numeric string like `"20"`)
- `render3d.camera_csv` (output/input CSV path for camera settings)

## Recommended usage

1) Inspect point cloud geometry

```bash
python inspect_ply_fields.py
```

2) Generate and validate camera CSV

```bash
python generate_dynamic_camera_csv.py --config spherical_unwrap.yaml
```

3) Run unwrap + rendering with imported camera CSV

```bash
python spherical_unwrap_frames_yaml.py
```

## Notes

- `sanity_check_camera_csv.py` is a standalone validation script (optional) - validation is now integrated into the generator.

- Re-run `generate_dynamic_camera_csv.py` whenever you change:
  - `progress.frames_to_have`
  - `progress.step_cols`
  - input PLY file
- `spherical_unwrap_frames_yaml.py` asserts that camera CSV row count and `frame_k` values match the exported frame plan.
