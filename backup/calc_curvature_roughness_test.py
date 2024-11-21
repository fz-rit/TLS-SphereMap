"""
Contributor: fzhcis@rit.edu
Version: 1.1
Last Updated: 11/19/2024
Description:
This script processes a point cloud to estimate curvature and roughness using GPU acceleration.
It includes functions for loading configuration, preprocessing point clouds, performing neighborhood searches,
estimating curvature and roughness, visualizing results, and exporting the processed data.
The script also monitors CPU and GPU memory usage during execution.
"""

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
import threading
from contextlib import contextmanager
from torch_cluster import radius_graph
import torch_geometric
from torch_scatter import scatter_mean


def load_config(json_path):
    """Load configuration from a JSON file."""
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config


def pad_neighbors(points, neighbors_list, max_neighbors):
    """Pad each neighborhood to the maximum number of neighbors for batch processing."""
    num_points = points.shape[0]
    padded_neighbors = torch.zeros((num_points, max_neighbors, 3), device=points.device)
    mask = torch.zeros((num_points, max_neighbors), device=points.device)

    for i, idx in enumerate(neighbors_list):
        num_neighbors = len(idx)
        if num_neighbors > max_neighbors:
            idx = idx[:max_neighbors]
            num_neighbors = max_neighbors
        padded_neighbors[i, :num_neighbors] = points[idx]
        mask[i, :num_neighbors] = 1

    return padded_neighbors, mask


def batch_neighborhood_search(points, radius, max_neighbors):
    """Perform neighborhood search using torch_cluster's radius_graph."""
    edge_index = radius_graph(points, r=radius, loop=False, max_num_neighbors=max_neighbors)
    num_points = points.shape[0]

    # Create a list of neighbor indices for each point
    neighbors_list = [[] for _ in range(num_points)]
    src_indices = edge_index[0]
    dst_indices = edge_index[1]

    for src, dst in zip(src_indices, dst_indices):
        neighbors_list[src.item()].append(dst.item())

    return neighbors_list


def estimate_curvature_roughness_batched(points, curvature_radius=0.05, roughness_radius=0.2, max_neighbors=10):
    """Estimate curvature and roughness for a batch of points with GPU-based neighborhood search."""
    points = torch.tensor(points, dtype=torch.float32, device='cuda')

    # Function to compute covariance matrices and eigenvalues
    def compute_covariances(points, radius):
        neighbors_list = batch_neighborhood_search(points, radius, max_neighbors)
        padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)

        centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
        centered_neighbors_masked = centered_neighbors * mask.unsqueeze(2)
        covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
        

        print("covariances shape:", covariances.shape)

        return covariances, mask, centered_neighbors_masked

    # Calculate curvature
    covariances, _, _ = compute_covariances(points, curvature_radius)

    # Compute eigenvalues for curvature
    eigenvalues, _ = torch.linalg.eigh(covariances)

    # Curvature calculation
    curvatures = eigenvalues[:, 0] / eigenvalues.sum(dim=1)

    # Calculate roughness
    covariances, mask, centered_neighbors_masked = compute_covariances(points, roughness_radius)

    # Compute eigenvalues and eigenvectors for roughness
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)

    # Normal vectors (corresponding to smallest eigenvalue)
    normal_vectors = eigenvectors[:, :, 0]

    # Roughness calculation
    roughness_numerators = torch.abs(torch.bmm(centered_neighbors_masked, normal_vectors.unsqueeze(2)).squeeze())
    roughness = roughness_numerators.sum(dim=1) / mask.sum(dim=1)

    return curvatures.cpu().numpy(), roughness.cpu().numpy()



def process_batches(df_filtered, curvature_radius, roughness_radius, max_neighbors, batch_size):
    """Process point cloud batches to calculate curvature and roughness."""
    all_points, all_curvatures, all_roughness = [], [], []
    num_points = df_filtered.shape[0]
    num_batches = int(np.ceil(num_points / batch_size))
    for i in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_points)
        df_batch = df_filtered.iloc[start_idx:end_idx]
        points = df_batch[['X', 'Y', 'Z']].to_numpy()
        curvatures, roughness = estimate_curvature_roughness_batched(points, curvature_radius, roughness_radius, max_neighbors=max_neighbors)
        all_points.append(points)
        all_curvatures.append(curvatures)
        all_roughness.append(roughness)
    return np.vstack(all_points), np.hstack(all_curvatures), np.hstack(all_roughness)


def visualize_results(all_points, all_curvatures, all_roughness):
    """Visualize curvature and roughness with Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colormap = plt.get_cmap('jet')

    # Visualize curvature
    all_curvatures_colors = colormap(all_curvatures)[:, :3]  # Get RGB values from colormap
    pcd.colors = o3d.utility.Vector3dVector(all_curvatures_colors)
    print("Displaying curvature visualization...")
    o3d.visualization.draw_geometries([pcd])

    # Visualize roughness
    all_roughness_colors = colormap(all_roughness)[:, :3]  # Get RGB values from colormap
    pcd.colors = o3d.utility.Vector3dVector(all_roughness_colors)
    print("Displaying roughness visualization...")
    o3d.visualization.draw_geometries([pcd])


def export_results(df_filtered_out,
                   all_curvatures,
                   all_roughness,
                   output_dir,
                   filename):
    """Append curvature and roughness to DataFrame and export."""
    df_filtered_out['curvature'] = all_curvatures
    df_filtered_out['roughness'] = all_roughness
    export_path = output_dir / f"{filename.stem}_curvature_roughness_test.txt"
    df_filtered_out.to_csv(export_path, sep=',', index=False)
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
    roughness_radius = config["roughness_radius"]
    max_neighbors = config["max_neighbors"]
    batch_size = config.get("batch_size", 5000)
    range1metres_min = config.get("range1metres_min", 2.25)
    range1metres_max = config.get("range1metres_max", 4.0)
    visualize = config.get("visualize", True)
    export = config.get("export", True)

    # Preprocess point cloud
    df_filtered = preprocess_point_cloud(filename, range1metres_min, range1metres_max)
    print(f"Filtered point cloud shape: {df_filtered.shape}")

    # Process in batches
    all_points, all_curvatures, all_roughness = process_batches(df_filtered, curvature_radius, roughness_radius, max_neighbors, batch_size)

    # Check and clean for NaN values in curvatures and roughness
    valid_mask_curv = check_and_clean_for_nans(all_curvatures)
    valid_mask_rough = check_and_clean_for_nans(all_roughness)
    valid_mask = valid_mask_curv & valid_mask_rough

    print(f"Number of valid points: {valid_mask.sum()}")
    all_points = all_points[valid_mask]
    df_filtered_out = df_filtered.iloc[valid_mask]
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
    if visualize:
        visualize_results(all_points, normalized_curvatures, normalized_roughness)

    # Export results
    if export:
        export_results(df_filtered_out,
                       normalized_curvatures,
                       normalized_roughness,
                       output_dir,
                       filename)


# Utility function for CPU memory monitoring
@contextmanager
def cpu_memory_monitoring():
    process = psutil.Process()
    peak_memory = [0]  # Using a list to allow modification within the nested function
    stop_event = threading.Event()

    def monitor_memory():
        while not stop_event.is_set():
            mem = process.memory_info().rss
            if mem > peak_memory[0]:
                peak_memory[0] = mem
            time.sleep(0.1)  # Adjust the sleep interval as needed

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    try:
        yield peak_memory
    finally:
        stop_event.set()
        monitor_thread.join()


# Utility function for GPU memory monitoring
@contextmanager
def gpu_memory_monitoring():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        pass  # No cleanup required for GPU monitoring


def main():
    # Start tracking time
    start_time = time.time()

    # Begin CPU and GPU memory monitoring
    with cpu_memory_monitoring() as peak_memory:
        with gpu_memory_monitoring():
            # Computational part
            config_path = Path('./input_params/calc_curvature_roughness_input_zmachine_test.json')
            config = load_config(config_path)
            calculate_curvature_and_roughness(config)

    # End tracking time
    elapsed_time = time.time() - start_time

    # Convert peak memory to MB
    peak_cpu_memory_mb = peak_memory[0] / (1024 * 1024)

    # Retrieve peak GPU memory usage
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_memory_mb = 0

    # Print results
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Peak CPU Memory Used: {peak_cpu_memory_mb:.2f} MB")
    print(f"Peak GPU Memory Used: {peak_gpu_memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
