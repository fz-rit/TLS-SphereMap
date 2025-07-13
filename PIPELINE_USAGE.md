# TLS-SphereMap Pipeline Usage Guide

## Overview
The TLS-SphereMap pipeline has been updated to support command-line arguments and automatic configuration loading. This makes it much easier to work with different datasets and configurations.

## Basic Usage

### Run with default steps
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic.json
```

### Run specific processing steps
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic.json --steps calc_geom_feature spherical_projection
```

### Available processing steps:
- `calc_geom_feature` - Calculate geometric features (curvature, anisotropy, planarity)
- `spherical_projection` - Project 3D points to 2D spherical images
- `spherical_back_projection` - Back-project 2D data to 3D points
- `convert_color_to_mask` - Convert color segmentation maps to masks
- `attach_segmap_to_points` - Attach segmentation data to point clouds

## Configuration Files

The pipeline automatically detects the dataset type based on the configuration file name or the `dataset` field in the config:

### Available datasets:
- **ForestSemantic**: `input_params/3D_to_2D_config_forestsemantic*.json`
- **Mangrove**: `input_params/3D_to_2D_config_mangrove*.json`
- **INLUT3D**: `input_params/3D_to_2D_config_inlut3d*.json`
- **Semantic3D**: `input_params/3D_to_2D_config_semantic3d*.json`

## Examples

### Process ForestSemantic data with all steps:
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic.json
```

### Process only spherical projection:
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic.json --steps spherical_projection
```

### Process INLUT3D data:
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_inlut3d.json
```

## What Changed

### Before (old method):
- Had to manually edit `config_loader.py` to uncomment the right config
- Could only run with hardcoded configuration
- No flexibility in choosing processing steps

### After (new method):
- Clean command-line interface with argparse
- Automatic config detection and loading
- Flexible step selection
- Environment variable-based config sharing between processes
- Better error handling and user feedback

## Technical Details

The pipeline now:
1. Uses argparse for command-line argument parsing
2. Loads configuration dynamically based on the provided path
3. Sets an environment variable (`TLS_CONFIG_PATH`) for subprocess communication
4. Automatically detects dataset type and applies appropriate processing functions
5. Provides clear error messages when configuration is not loaded properly

## Troubleshooting

If you see an error like "Configuration not loaded", make sure to:
1. Run the script through `run_3d_to_2d_pipeline.py` (not individual processing modules)
2. Provide a valid config file path
3. Ensure the config file exists and is properly formatted

### Valid command:
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_forestsemantic.json
```

### Invalid command (will show error):
```bash
python -m processing.spherical_projection  # Don't run modules directly
```
