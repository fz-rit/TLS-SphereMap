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
- `refine_pcd_labels.py` - Refine point cloud labels based on segmentation
- `refine_2D_mask.py` - Refine 2D segmentation masks from refined point cloud labels

## Configuration Files

The pipeline automatically detects the dataset type based on the configuration file name or the `dataset` field in the config:

### Available datasets:
- **ForestSemantic**: `input_params/3D_to_2D_config_forestsemantic*.json`
- **Mangrove**: `input_params/3D_to_2D_config_mangrove*.json`
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

### Process Semantic3D data:
```bash
python run_3d_to_2d_pipeline.py --config input_params/3D_to_2D_config_Semantic3D.json
```


## Troubleshooting

If you see an error like "Configuration not loaded", make sure to:
1. Run the script through `run_3d_to_2d_pipeline.py` (not individual processing modules)
2. Provide a valid config file path
3. Ensure the config file exists and is properly formatted

