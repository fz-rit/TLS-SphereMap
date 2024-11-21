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
- First, calculate the normals of the point cloud.
    - Make a copy of the `calc_pt_cloud_normal_inputs.json` file and adjust the input output paths in the json file
    - Adjust the path pointing to the new json file in `calc_pt_cloud_normal.py`
    - run:  
        ```
        python calc_pt_cloud_normal.py
        ```
- Second, generate the unwrapped images. Adjust input paths in the jupyter notebook `Unwrap_TLS_to_2d_v1.4.ipynb` and then run the whole notebook.