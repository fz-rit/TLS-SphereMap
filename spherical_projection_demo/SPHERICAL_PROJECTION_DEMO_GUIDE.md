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
  - Exports 2D unwrap frames with axes showing azimuth/zenith angles.
  - Grid dimensions computed adaptively from angular ranges and resolution.
  - Renders 3D screenshots with two fixed views:
    - Side view (front view, x-z plane)
    - Top-down view
  - Point cloud colors match the 2D projection colormap values.

## Required config keys (in `spherical_unwrap.yaml`)

- `grid.angular_resolution` - Angular resolution in degrees (e.g., 0.25 for TLS)
- `progress.step_degrees` - Angular step between frames in degrees (e.g., 0.5)
- `progress.frames_to_have` (`"max"` or numeric string like `"20"`)
- `progress.tls_dual_scan` - If `true`, simulates TLS scanning pattern where opposite azimuths (180° apart) are captured simultaneously. Physical rotation is 180° which covers the full 360° sphere.
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

## TLS Dual-Scan Pattern

When `progress.tls_dual_scan: true`, the animation simulates the actual TLS scanning pattern where the scanner captures opposite azimuths simultaneously:
- At rotation angle 0°: captures both 0° and 180°
- At rotation angle 0.25°: captures both 0.25° and 180.25°
- At rotation angle 0.5°: captures both 0.5° and 180.5°
- etc.

**Important**: With dual-scan enabled, a **180° physical rotation covers the full 360° sphere** because both front and back hemispheres are captured simultaneously. The frame sequence runs from 0° to 180° rotation, not 0° to 360°.

This creates a "reveal from opposite sides" effect that matches how terrestrial laser scanners actually acquire data, rather than a simple left-to-right sequential scan.

## Adaptive Grid Dimensions

Grid dimensions (width × height) are computed automatically from:
- Angular ranges: `az_max - az_min` and `ze_max - ze_min`
- Angular resolution: `grid.angular_resolution` (default 0.25°)

For example, with azimuth 0-360° and zenith 0-135° at 0.25° resolution:
- Width = 360 / 0.25 = 1440 columns
- Height = 135 / 0.25 = 540 rows

The 2D images display axes with angular tick labels (degrees) for easy interpretation.

## Frame Stepping

`progress.step_degrees` controls the angular advancement between animation frames. For example:
- `step_degrees: 0.5` with `angular_resolution: 0.25` → advance by 2 columns per frame
- `step_degrees: 1.0` with `angular_resolution: 0.25` → advance by 4 columns per frame

With `tls_dual_scan: true`, the frame sequence spans 0-180° rotation (half the azimuth range) since the opposite hemisphere is captured simultaneously.
