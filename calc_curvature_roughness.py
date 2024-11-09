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
from matplotlib import pyplot as plt

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

def batch_neighborhood_search(points, radius, max_neighbors, device):
    """Perform neighborhood search on the GPU in batches."""
    points = points.to(device)
    dist_matrix = torch.cdist(points, points)
    # neighbors_list = [(dist_matrix[i] <= radius).nonzero(as_tuple=True)[0] for i in range(len(points))]
    neighbors_list = []

    for i in range(len(points)):
        # Find neighbors within the radius
        neighbors = (dist_matrix[i] <= radius).nonzero(as_tuple=True)[0]

        # Limit the number of neighbors to max_neighbors
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]

        neighbors_list.append(neighbors)
    return neighbors_list

def estimate_curvature_roughness_batched(points, radius=0.05, max_neighbors=10):
    """Estimate curvature for a batch of points with GPU-based neighborhood search."""
    points = torch.tensor(points, dtype=torch.float32, device='cuda')
    # Calculate curvature
    neighbors_list = batch_neighborhood_search(points, radius, max_neighbors, device='cuda')
    padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)
    
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    
    curvatures = eigenvalues[:, 0] / eigenvalues.sum(dim=1)

    # Calculate roughness
    neighbors_list = batch_neighborhood_search(points, radius=0.15, max_neighbors=max_neighbors, device='cuda')
    padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)
    
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    
    normal_vectors = eigenvectors[:, :, 0]  # Normal vectors (corresponding to smallest eigenvalue)
    roughness = torch.abs((centered_neighbors * mask.unsqueeze(2)).matmul(normal_vectors.unsqueeze(2)).squeeze()).mean(dim=1)

    return curvatures.cpu().numpy(), roughness.cpu().numpy()


def process_batches(dfs, curvature_radius, max_neighbors):
    """Process point cloud batches to calculate curvature and roughness."""
    all_points, all_curvatures, all_roughness = [], [], []
    for df_batch in tqdm(dfs, desc="Processing Batches", unit="batch"):
        points = df_batch[['X', 'Y', 'Z']].to_numpy()
        curvatures, roughness = estimate_curvature_roughness_batched(points, radius=curvature_radius, max_neighbors=max_neighbors)
        all_points.append(points)
        all_curvatures.append(curvatures)
        all_roughness.append(roughness)
    return np.vstack(all_points), np.hstack(all_curvatures), np.hstack(all_roughness)

def visualize_results(all_points, all_curvatures, all_roughness):
    """Visualize curvature and roughness with Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colormap = plt.get_cmap('jet')
    all_curvatures_colors = colormap(all_curvatures)[:, :3]  # Get RGB values from colormap
    all_roughness_colors = colormap(all_roughness)[:, :3]  # Get RGB values from colormap
    
    # Visualize curvature
    # Apply a colormap (e.g., jet colormap)

    pcd.colors = o3d.utility.Vector3dVector(all_curvatures_colors)
    print("Displaying curvature visualization...")
    o3d.visualization.draw_geometries([pcd])

    # Visualize roughness
    pcd.colors = o3d.utility.Vector3dVector(all_roughness_colors)
    print("Displaying roughness visualization...")
    o3d.visualization.draw_geometries([pcd])

def export_results(df_filtered, all_curvatures, 
                   all_roughness, 
                   output_dir):
    """Append curvature and roughness to DataFrame and export."""
    df_filtered['curvature'] = all_curvatures
    df_filtered['roughness'] = all_roughness
    export_path = output_dir / "processed_point_cloud.txt"
    df_filtered[['X', 'Y', 'Z', 'curvature', 'roughness']].to_csv(export_path, sep='\t', index=False)
    print(f"Exported point cloud with curvature and roughness to {export_path}")

def check_and_clean_for_nans(all_curvatures):
    # Check for NaN values in curvatures and roughness
    valid_mask = np.ones_like(all_curvatures, dtype=bool)
    if np.isnan(all_curvatures).any():
        print("Warning: NaN values detected in curvature/roughness calculation. Removing NaNs for histogram.")
        valid_mask = ~np.isnan(all_curvatures)
    
    if len(all_curvatures) == 0:
        raise ValueError("Error: No valid curvature or roughness values to plot.")
        
    return valid_mask

def calculate_curvature_and_roughness(config):
    """Process the point cloud, estimate curvature and roughness, and combine results."""
    filename = Path(config["filename"])
    output_dir = Path(config["output_dir"])
    curvature_radius = config["curvature_radius"]
    max_neighbors = config["max_neighbors"]
    range1metres_min = config.get("range1metres_min", 2.25)
    range1metres_max = config.get("range1metres_max", 4.0)

    # Preprocess point cloud
    df_filtered = preprocess_point_cloud(filename, range1metres_min, range1metres_max)
    print(f"Filtered point cloud shape: {df_filtered.shape}")

    # Check the number of points and decide whether to batch or process directly
    num_points = df_filtered.shape[0]
    if num_points > 50000:
        print("*********Large dataset detected. Processing in batches...*********")
        df_grouped = df_filtered.groupby(pd.cut(df_filtered['azimuth'], 10))
        dfs = [group for _, group in df_grouped]
        all_points, all_curvatures, all_roughness = process_batches(dfs, curvature_radius, max_neighbors)
    else:
        print("*********Small dataset detected. Processing all at once...*********")
        all_points = df_filtered[['X', 'Y', 'Z']].to_numpy()
        all_curvatures, all_roughness = estimate_curvature_roughness_batched(all_points, radius=curvature_radius, max_neighbors=max_neighbors)

    # Check for NaN values in curvatures and roughness
    valid_mask_curv = check_and_clean_for_nans(all_curvatures)
    valid_mask_rough = check_and_clean_for_nans(all_roughness)
    valid_mask = valid_mask_curv & valid_mask_rough
    valid_mask = valid_mask_curv

    print(f"Number of valid points: {valid_mask.sum()}")
    all_points = all_points[valid_mask]
    df_filtered_out = df_filtered[valid_mask]
    all_curvatures = all_curvatures[valid_mask]
    all_roughness = all_roughness[valid_mask]

    

    # Generate histograms for curvature and roughness
    get_histogram(all_curvatures, output_dir, title="Curvature", saveflag=config["histogram_saveflag"])
    get_histogram(all_roughness, output_dir, title="Roughness", saveflag=config["histogram_saveflag"])


    # Normalize curvature and roughness for visualization
    normalized_curvatures = (all_curvatures - all_curvatures.min()) / (all_curvatures.max() - all_curvatures.min())
    normalized_roughness = (all_roughness - all_roughness.min()) / (all_roughness.max() - all_roughness.min())

    get_histogram(normalized_curvatures, output_dir, title="Normalized Curvature", saveflag=config["histogram_saveflag"])
    get_histogram(normalized_roughness, output_dir, title="Normalized Roughness", saveflag=config["histogram_saveflag"])

    # Visualize results
    visualize_results(all_points, normalized_curvatures, normalized_roughness)

    # Export results
    export_results(df_filtered_out, 
                   normalized_curvatures, 
                   normalized_roughness, 
                   output_dir)

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
    
    calculate_curvature_and_roughness(config)
    
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
