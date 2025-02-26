"""
calc_curvature_roughness_kdtree.py
This script processes a point cloud to estimate curvature and roughness using Open3D’s KD-Tree 
(`KDTreeFlann`) for efficient neighborhood search on CPU, removing GPU acceleration.

Contributor: fzhcis@rit.edu
Version: 2.0
Last Updated: 02/18/2025
"""

import open3d as o3d
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
from preprocess_point_cloud import read_raw_point_cloud
from plot_tools import get_vector_histogram
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Any
from config_loader import CONFIG
import torch


def search_and_pad_neighbors(points_xyz: np.ndarray, radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perform neighborhood search using Open3D’s KD-Tree (`KDTreeFlann`) and pad neighbors in one function.

    Args:
        points_xyz (np.ndarray): Nx3 array of 3D points.
        radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Padded neighbors and valid mask.
    """
    num_points = len(points_xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Pre-allocate arrays
    padded_neighbors = np.zeros((num_points, max_neighbors, 3))
    mask = np.zeros((num_points, max_neighbors))

    for i in tqdm(range(num_points), desc="Searching & Padding Neighbors"):
        [_, idx, _] = kdtree.search_radius_vector_3d(points_xyz[i], radius)
        idx = np.array(idx[:max_neighbors])  # Limit to max_neighbors

        num_neighbors = len(idx)
        if num_neighbors > 0:
            padded_neighbors[i, :num_neighbors] = points_xyz[idx]
            mask[i, :num_neighbors] = 1  # Mark valid neighbors

    return padded_neighbors, mask



def calculate_neighbor_eigens(points_xyz: np.ndarray, 
                              nn_radius: float, 
                              max_neighbors: int
                             ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Calculate eigenvalues and eigenvectors for each point's neighborhood using PyTorch.
    
    Args:
        points_xyz (np.ndarray): Array of points (N,3).
        nn_radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to consider.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - Eigenvalues: (N, 3)
            - Eigenvectors: (N, 3, 3)
            - A tuple of (centered_neighbors, mask)
    """
    # Set device (cuda if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert points to a torch tensor on the proper device
    # make a copy of the points_xyz without changing the original
    points_xyz_tensor = torch.tensor(points_xyz, dtype=torch.float32, device=device)
    
    # Run the neighborhood search and padding (assumed to be defined elsewhere)
    padded_neighbors, mask = search_and_pad_neighbors(points_xyz_tensor, nn_radius, max_neighbors)
    
    # Ensure mask is float for correct arithmetic
    mask = mask.float()
    
    # Compute centroids for each point's neighborhood
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    centroids = padded_neighbors.sum(dim=1) / valid_counts
    
    # Compute centered neighbors
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    
    # Compute covariance matrices per point
    # Note: Multiply centered_neighbors by mask.unsqueeze(2) so that invalid neighbors contribute 0.
    #   centered_neighbors: (N, max_neighbors, 3)
    #   centered_neighbors.transpose(1,2): (N, 3, max_neighbors)
    #   Their matmul produces (N, 3, 3)
    denominator = (mask.sum(dim=1) - 1).clamp(min=1e-6).view(-1, 1, 1)
    covariances = torch.matmul(centered_neighbors.transpose(1, 2),
                               centered_neighbors * mask.unsqueeze(2)) / denominator
    
    # Compute eigenvalues and eigenvectors (each covariance matrix is 3x3)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    
    # Debug prints for shapes
    print(f"Shape of padded_neighbors: {padded_neighbors.shape}")  # (N, max_neighbors, 3)
    print(f"Shape of mask: {mask.shape}")                          # (N, max_neighbors)
    print(f"Shape of centered_neighbors: {centered_neighbors.shape}")# (N, max_neighbors, 3)
    print(f"Shape of covariances: {covariances.shape}")            # (N, 3, 3)
    print(f"Shape of eigenvalues: {eigenvalues.shape}")            # (N, 3)
    print(f"Shape of eigenvectors: {eigenvectors.shape}")          # (N, 3, 3)
    
    return eigenvalues, eigenvectors, (centered_neighbors, mask)




def estimate_curvature_roughness(points_xyz: np.ndarray, curvature_radius: float, roughness_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate curvature and roughness for all points.

    Args:
        points_xyz (np.ndarray): Nx3 array of points.
        curvature_radius (float): Radius for curvature estimation.
        roughness_radius (float): Radius for roughness estimation.
        max_neighbors (int): Maximum number of neighbors.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Curvature and roughness arrays.
    """
    # Curvature estimation
    curv_eigenval, rough_eigenvec, (centered_neighbors, mask) = calculate_neighbor_eigens(points_xyz, curvature_radius, max_neighbors)
    denominator = np.sum(curv_eigenval, axis=1)
    
    # Create a mask where there are no valid neighbors (denominator == 0)
    no_neighbors_mask = (denominator == 0)

    # Assign a very small value where there are no valid neighbors
    denominator = np.where(no_neighbors_mask, 1, denominator)  # Avoid division by zero
    curvatures = curv_eigenval[:, 0] / denominator

    # Assign a small value (e.g., 1e-6) where no valid neighbors exist
    curvatures[no_neighbors_mask] = 1e-6 

    print(f"Shape of rough_eigenvec: {rough_eigenvec.shape}")  # (N, 3, 3)
    normal_vectors = rough_eigenvec[:, 0, :]
    # Print shape debugging info
    print(f"Shape of centered_neighbors: {centered_neighbors.shape}")  # (N, max_neighbors, 3)
    print(f"Shape of mask: {mask.shape}")  # (N, max_neighbors)
    print(f"Shape of normal_vectors before expansion: {normal_vectors.shape}")  # (N, 3)

    # Expand normal_vectors to match centered_neighbors
    normal_vectors = np.repeat(normal_vectors[:, np.newaxis, :], max_neighbors, axis=1)

    print(f"Shape of normal_vectors after expansion: {normal_vectors.shape}")  # (N, max_neighbors, 3)


    roughness = np.mean(np.abs(np.einsum('nij,nj->ni', centered_neighbors * mask[:, :, None], normal_vectors)), axis=1)

    return curvatures, roughness


def process_point_cloud(config: Dict[str, Any]):
    """Load, process, and export curvature and roughness.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    global_params = config["global"]
    params = config["calc_curvature_roughness"]
    output_dir = Path(global_params["output_dir"])
    input_file_stem = global_params['input_file_stem']
    input_path = Path(output_dir / f"{input_file_stem}_filtered_normaled.txt")

    curvature_radius = params["neighbor_radius"]
    roughness_radius = params["neighbor_radius"]
    max_neighbors = params["max_neighbors"]

    df = pd.read_csv(input_path, sep=',')
    points_xyz = df[['X', 'Y', 'Z']].to_numpy()

    print(f"Processing {len(points_xyz)} points...")
    start_time = time.time()

    curvatures, roughness = estimate_curvature_roughness(points_xyz, curvature_radius, roughness_radius, max_neighbors)

    df['curvature'] = curvatures
    df['roughness'] = roughness
    df.to_csv(output_dir / f"{input_file_stem}_processed.csv", index=False)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {output_dir}/{input_file_stem}_processed.csv")


def main():
    """Main function to execute the point cloud processing script."""
    process_point_cloud(CONFIG)


if __name__ == "__main__":
    main()
