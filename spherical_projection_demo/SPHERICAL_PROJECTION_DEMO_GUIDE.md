# Spherical Projection Demo Guide

## Folder structure

```text
spherical_projection_demo/
├─ inspect_ply_fields.py
├─ spherical_unwrap_frames_yaml.py
├─ spherical_unwrap.yaml
└─ SPHERICAL_PROJECTION_DEMO_GUIDE.md
```

## Script roles

- `inspect_ply_fields.py`
  - Reads `input_ply` from `spherical_unwrap.yaml`.
  - Prints scalar field statistics.

- `spherical_unwrap_frames_yaml.py`
  - Reads `spherical_unwrap.yaml`.
  - Exports 2D unwrap frames.
  - Renders 3D screenshots with two fixed views:
    - Side view (front view, x-z plane)
    - Top-down view
  - Point cloud colors match the 2D projection colormap values.

## Required config keys (in `spherical_unwrap.yaml`)

- `progress.step_cols`
- `progress.frames_to_have` (`"max"` or numeric string like `"20"`)
- `render3d.side_subdir` (output folder for side views)
- `render3d.top_subdir` (output folder for top views)

## Usage

1) Inspect point cloud fields (optional)

```bash
python inspect_ply_fields.py
```

2) Run unwrap and rendering

```bash
python spherical_unwrap_frames_yaml.py
```

## Output

- `frames_2d/` - 2D spherical unwrap frames
- `frames_3d_side/` - 3D side view screenshots
- `frames_3d_top/` - 3D top-down view screenshots
