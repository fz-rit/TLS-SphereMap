# **Spherical Projection for Simplifying 3D Point Cloud Annotation**  

Labeling 3D point clouds can be time-consuming and challenging, especially in complex forest environments. This repository offers a way to make the process **more efficient and accessible** by converting terrestrial LiDAR data into **2D spherical projection images**, where each pixel represents scalar values like **intensity, range, curvature, and roughness**.  

### **How It Works:**  
1. **Convert 3D point clouds** into **spherical projection images** based on elevation and azimuth angles.  
2. **Use familiar 2D annotation tools** to segment objects (e.g., roots, stems, and canopy in forest scenes), reducing the complexity of direct 3D labeling.  
3. **Map the labeled masks back to the 3D point cloud**, maintaining structural accuracy while making annotation more manageable.  

### **Why It Matters:**  
- **A Practical Approach** – Adapts a 2D workflow to help with a common challenge in 3D annotation.  
- **More Efficient Annotation** – Speeds up the labeling process, especially for large-scale datasets.  
- **Supports 3D Scene Understanding** – Helps generate training data for deep learning in **natural, unstructured environments** where labeled datasets are still limited.  


While the test datasets focus on forest scenes—Harvard Forest and Palau Mangrove Roots—this method is versatile and can be applied to a wide range of terrestrial LiDAR survey applications. If you're working with TLS point clouds and looking for a more efficient and intuitive way to annotate them, this tool might be a useful addition to your workflow! 🤞

## Table of Contents
- [Introduction](#spherical-projection-for-simplifying-3d-point-cloud-annotation)
- [Installation](#installation)
- [Usage](#usage)
- [Intermediary Results](#intermediary-results)
- [Segmented Point Clouds](#segmented-point-clouds)
- [To Do](#to-do)
---
# Installation
```bash
# create a new conda environment
conda create -n tls_env python=3.9
conda activate tls_env

# Install dependencies with pip
pip install open3d
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-image
pip install laspy
pip install plyfile

# To solve the open3d `Segmentation fault` issue.
pip install numpy==1.26.4


# Install torch-geometric related dependencies, suppose you have PyTorch CUDA version: 12.1 and PyTorch version: 2.4.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric
pip install seaborn

```

# Usage
1) Prepare input files, current supported input formats:
    - ".txt" - from Palau Mongrove scans, derived from .gbl files using 'CBL_GBL_Processing_UMB.py', scalar fields apart from x/y/z: 
        1. zenith angle (0-135) 
            - Definition: the angle between the vertical line (directly above the observer) and an observed object in the sky. A zenith angle of 0° means the object is directly overhead. However, notice that LiDAR was set upside down for scanning the roots, so 0° means the point is right under the LiDAR, and 135° means the point that was 45° above the lidar.
                - The scan pattern of the TLS for mangrove roots:
                    - <img src="examples/tls_scan_angle_pattern.png" alt="tls_scan_angle_pattern" width="400"/>
            
        2. azimuth angle (0-360°)
        3. range1meter (0-R; usually R<100)
        4. intensity (aka remission; 0-4000)
        5. return number (1 or 2)
    - ".las" - from Harvard Forest scans, scalar fields apart from x/y/z:
        1. intensity (aka remission; 0-4000)
        2. return number (1 or 2)
    - ".bin" - from SemanticKitti dataset, scalar fields apart from x/y/z:
        1. intensity (aka remission; 0.00 - 0.99)
2) Calculate the spherical projection images from the point cloud:
    - Change the `CONFIG_PATH` in the config_loader.py to the desired config file, e.g., `3D_to_2D_config_mangrove_roots.json`
    - Adjust the paths in the `3D_to_2D_config_mangrove_roots.json` file
    - run:  `python run_3d_to_2d_pipeline.py`
        - outputs include unwrapped 2D images and a point cloud .txt file with fields:
            - 'X': 'float32',
            - 'Y': 'float32',
            - 'Z': 'float32',
            - 'Intensity': 'uint16',
            - 'Return Number': 'uint8',
            - 'azimuth': 'float32',
            - 'zenith': 'float32',
            - 'elevation': 'float32',
            - 'range1metres': 'float32',
            - 'nx': 'float32',
            - 'ny': 'float32',
            - 'nz': 'float32'
            - 'curvature': 'float32'
            - 'roughness': 'float32'
3) Manually label the 2D unwrapped images to get segmentation maps in RGB, e.g., `seg_map_33_01.png`.
    - refer to the [`data_annotation_guidline.md`](data_annotation/data_annotation_guidline.md) for more details.
4) Generate class ID based segmentaion map and then attach the class id and color to the point cloud:
    - refer to the [`seg_map_tools_readme.md`](data_annotation/seg_map_tools/seg_map_tools_readme.md) for more details.
5) [Optional] Co-register multiple scans of point clouds:
    - Use [`GlobalMatch`](https://github.com/zexinyang/GlobalMatch) to get the rigit transformation matrices (use original point clouds as inputs).
    - Load the point clouds and apply the transformation matrices.

## Intermediary Results

### 2D Spherical Projection Images
| Harvard Forest Intensity Map | Mangrove Roots Intensity Map |
|---------------------------|-------------------------------------|
| ![Harvard Forest Intensity Map](examples/Intensity%20Map%20%28adjusted%29_33_01.png) | ![Mangrove Roots Intensity Map](examples/Intensity%20Map%20%28adjusted%29_UMBCBL009_1830507489.png) |

| Harvard Forest Pseudo-RGB_Roughness-Intensity-Range | Mangrove Roots Pseudo-RGB_Roughness-Intensity-Range |
|---------------------------------|-------------------------------------------|
| ![Harvard Forest Pseudo-RGB_Roughness-Intensity-Range-33_01](examples/Pseudo-RGB_Roughness-Intensity-Range-33_01.png) | ![Mangrove Roots Pseudo-RGB_Roughness-Intensity-Range-UMBCBL009_1830507489](examples/Pseudo-RGB_Roughness-Intensity-Range-UMBCBL009_1830507489.png) |


### 2D Segmentation Maps (Annotated in Photoshop)
| Harvard Forest Segmentation Map | Mangrove Segmentation Map | 
|---------------------------|-------------------------------------|
| ![Harvard Forest Segmentation Map](examples/seg_map_harvard_33_02.png) | ![Mangrove Segmentation Map](examples/seg_map_ALRSET1_7489.png) |

| Harvard Forest Grayscale Segmentation Map | Mangrove Grayscale Segmentation Map | 
|---------------------------------|-------------------------------------------|
| ![Harvard Forest Grayscale Segmentation Map](examples/seg_map_harvard_33_02_mask_visualized.png) | ![Mangrove Grayscale Segmentation Map](examples/seg_map_ALRSET1_7489_mask_visualized.png) |

## Segmented Point Clouds
| Harvard Forest Segmented Point Cloud | Mangrove Segmented Point Cloud | 
|---------------------------------|-------------------------------------------|
| ![Harvard Forest Segmented Point Cloud](examples/HarvardForestSegmentPC.gif) | ![Mangrove Segmented Point Cloud](examples/MangroveRootsSegmentPC.gif) | 

<!-- (Note: URL-encode spaces (%20) and parentheses (%28, %29)) -->

## To Do
- [ ] Fix the obvious pattern cause by batch processing in Curvature map and Roughness map. (possibly by introducing the [`KDTree`](calc_curvature_roughness_kdtree.py))