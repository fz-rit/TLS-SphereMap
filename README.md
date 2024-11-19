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

# To solve the open3d Segmentation fault issue.
pip install numpy==1.26.4
```