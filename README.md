# tls_point_segmentation
 TLS point cloud segmentation through 2D segmentation on angularly-unwrapped maps


# Installation
```
# create a new conda environment
conda create -n tls_env python=3.9
conda activate tls_env

# Install dependencies with pip
pip install open3d
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-image
pip install laspy

# To solve the open3d Segmentation fault issue.
pip install numpy==1.26.4

# Install torch-geometric related dependencies, suppose you have PyTorch CUDA version: 12.1 and PyTorch version: 2.4.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric

```

# Usage
1) Prepare input files, current supported input formats:
    - ".txt" - from Palau Mongrove scans, derived from .gbl files using, scalar fields apart from x/y/z: 
        1. zenith angle (0-135) 
            - Explanation: zenith angle is defined as the angle between the vertical line (directly above the observer) and an observed object in the sky. A zenith angle of 0° means the object is directly overhead. However, notice that LiDAR was set upside down for scanning the roots, so 0 degree means the point is right under the LiDAR, and 135 degree means the point that was 45 degree above the lidar.
        2. azimuth angle (0-360)
        3. range1meter (0-R; usually R<100)
        4. intensity (aka remission; 0-4000)
        5. return number (1 or 2)
    - ".las" - from Harvard Forest scans, scalar fields apart from x/y/z:
        1. intensity (aka remission; 0-4000)
        2. return number (1 or 2)
    - ".bin" - from SemanticKitti dataset, scalar fields apart from x/y/z:
        1. intensity (aka remission; 0.00 - 0.99)
1) Calculate the normals of the point cloud.
    - Make a copy of the `calc_pt_cloud_normal_inputs.json` file and adjust the input output paths in the json file
    - Adjust the path pointing to the new json file in `calc_pt_cloud_normal.py`
    - run:  `python calc_pt_cloud_normal.py`
        - output a txt file with fields:
            - 'X': 'float32',
            - 'Y': 'float32',
            - 'Z': 'float32',
            - 'Intensity': 'uint16',
            - 'Return Number': 'uint8',
            - 'azimuth': 'float32',
            - 'zenith': 'float32',
            - 'range1metres': 'float32',
            - 'x_pix': 'uint16',
            - 'y_pix': 'uint16',
            - 'nx': 'float32',
            - 'ny': 'float32',
            - 'nz': 'float32'
2) [Optional] Generate treeiso single-tree segmentation labels, 
    - Make a copy of the `isolate_trees_inputs_amiri.json` file and adjust the input output paths in the json file 
    - Adjust the path to the new json file in `isolate_trees.py` and run it with  `python isolate_trees.py`
3) Generate curvature and roughness at the points.
    - Make a copy of the `calc_curvature_roughness_input_zmachine.json` file and adjust the input output paths in the json file
    - Adjust the path pointing to the new json file in `calc_curvature_roughness.py`
    - run:  `python calc_curvature_roughness.py`
        - output a txt file with fields:
            - 'X': 'float32',
            - 'Y': 'float32',
            - 'Z': 'float32',
            - 'Intensity': 'uint16',
            - 'Return Number': 'uint8',
            - 'azimuth': 'float32',
            - 'zenith': 'float32',
            - 'range1metres': 'float32',
            - 'x_pix': 'uint16',
            - 'y_pix': 'uint16',
            - 'nx': 'float32',
            - 'ny': 'float32',
            - 'nz': 'float32'
            - 'curvature': 'float32'
            - 'roughness': 'float32'
3) Generate the unwrapped images. Adjust input paths in the jupyter notebook `Unwrap_TLS_to_2d_v1.5.ipynb` and then run the whole notebook.
4) Manually label the 2D unwrapped images to get segmentation maps in RGB and in grayscale, e.g., `seg_map_manual_33_01.png` and `seg_map_manual_33_01.tif`.
5) Attach the color and label of the segmentation map to the point cloud:
    - Make a copy of the `attach_color_to_points_inputs.json` file and then modify the paths
    - run: `python attach_segmap_to_points.py`
6) Co-register multiple scans of point clouds:
    - Use GlobalMatch to get the rigit transformation matrices.
    - Load the point clouds and apply the transformation matrices.