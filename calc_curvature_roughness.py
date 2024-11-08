import open3d as o3d
import numpy as np
import pandas as pd
import json
import time
import torch
import psutil
from pathlib import Path
from tqdm import tqdm
from preprocess_point_cloud import preprocess_point_cloud
from plot_tools import get_histogram

def load_config(json_path):
    """Load configuration from a JSON file."""
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config

def pad_neighbors(points, neighbors_list, max_neighbors):
    """Pad each neighborhood to the maximum number of neighbors for batch processing."""
    padded_neighbors = []
    mask = []
    
    for idx in neighbors_list:
        neighbors = points[idx]
        if len(neighbors) < max_neighbors:
            padded = torch.cat([neighbors, torch.zeros((max_neighbors - len(neighbors), 3), device=points.device)])
            mask.append(torch.cat([torch.ones(len(neighbors), device=points.device), torch.zeros(max_neighbors - len(neighbors), device=points.device)]))
        else:
            padded = neighbors[:max_neighbors]
            mask.append(torch.ones(max_neighbors, device=points.device))
        
        padded_neighbors.append(padded)
    
    return torch.stack(padded_neighbors), torch.stack(mask)

def estimate_curvature_batched(pcd, radius=0.05, max_neighbors=10):
    """Estimate the curvature of a point cloud in a batched manner."""
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    neighbors_list = []
    
    for point in tqdm(points.cpu().numpy(), desc="Finding Neighbors"):
        _, idx, _ = pcd_tree.search_radius_vector_3d(point, radius)
        neighbors_list.append(idx)
    
    padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)
    
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    eigenvalues, _ = torch.linalg.eigh(covariances)
    
    curvatures = eigenvalues[:, 0] / eigenvalues.sum(dim=1)
    
    return curvatures.cpu().numpy()

def calculate_curvature(config):
    """Process the point cloud and estimate curvature."""
    filename = Path(config["filename"])
    output_dir = Path(config["output_dir"])
    curvature_radius = config["curvature_radius"]
    max_neighbors = config["max_neighbors"]
    range1metres_min = config.get("range1metres_min", 2.25)  # Default to 2.25 if not specified
    range1metres_max = config.get("range1metres_max", 4.0)    # Default to 4.0 if not specified

    
    # Preprocess point cloud
    df_filtered = preprocess_point_cloud(filename, range1metres_min, range1metres_max)
    print(f"Filtered point cloud shape: {df_filtered.shape}")
    points = df_filtered[['X', 'Y', 'Z']].to_numpy()
    
    # Load points into Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate curvature
    curvatures = estimate_curvature_batched(pcd, radius=curvature_radius, max_neighbors=max_neighbors)
    
    # Check for NaN values in curvatures
    if np.isnan(curvatures).any():
        print("Warning: NaN values detected in curvature calculation. Removing NaNs for histogram.")
        curvatures = curvatures[~np.isnan(curvatures)]
    
    if len(curvatures) == 0:
        print("Error: No valid curvature values to plot.")
        return

    # Generate histogram
    get_histogram(curvatures, output_dir, title=config["histogram_title"], saveflag=config["histogram_saveflag"])
    
    # Assign curvatures as colors for visualization
    pcd.colors = o3d.utility.Vector3dVector(np.tile(curvatures[:, None], (1, 3)))
    o3d.visualization.draw_geometries([pcd])

def main():
    # Start tracking time
    start_time = time.time()
    
    # Start tracking memory usage
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Start tracking GPU memory
    if torch.cuda.is_available():
        start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    else:
        start_gpu_memory = 0
    
    config_path = Path('/home/felix/mylab/tls_point_segmentation/input_params/calc_curvature_roughness_input.json')
    config = load_config(config_path)
    
    calculate_curvature(config)
    
    # End tracking time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # End tracking memory usage
    end_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    memory_used = end_memory - start_memory
    
    # End tracking GPU memory
    if torch.cuda.is_available():
        end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_memory_used = end_gpu_memory - start_gpu_memory
    else:
        gpu_memory_used = 0
    
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"CPU Memory Used: {memory_used:.2f} MB")
    print(f"GPU Memory Used: {gpu_memory_used:.2f} MB")

if __name__ == "__main__":
    main()
