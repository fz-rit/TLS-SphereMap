"""
calc_curvature_roughness.py
This module provides a comprehensive workflow for processing and analyzing 3D point cloud data
with GPU-accelerated neighborhood searches for estimating curvature and roughness. It 
demonstrates point cloud filtering, batched neighborhood searches, curvature and roughness 
computation, data visualization, and export functionalities.

Classes:
    None

Functions:
    load_config(json_path: str) -> Dict[str, Any]
        Loads configuration parameters from a specified JSON file.
    pad_neighbors(points: torch.Tensor, neighbors_list: List[torch.Tensor], max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]
        Ensures each point has a uniform number of neighbors for batch processing by padding.
    batch_neighborhood_search(points: torch.Tensor, radius: float, max_neighbors: int, device: str) -> List[torch.Tensor]
        Performs GPU-accelerated search for neighbors within a specified radius for each point.
    calculate_neighbor_eigens(points: np.ndarray, nn_radius: float, max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Calculates eigenvalues and eigenvectors for each point's neighborhood.
    estimate_curvature_roughness_batched(points: np.ndarray, curvature_radius: float, roughness_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]
        Estimates curvature and roughness for all points in a batched manner using GPU.
    process_batches(dfs: List[pd.DataFrame], curvature_radius: float, roughness_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Handles large datasets by splitting them into batches for curvature and roughness estimation.
    interactive_visualize_pcd(all_points: np.ndarray, all_curvatures: np.ndarray, all_roughness: np.ndarray, curvature_radius: float, roughness_radius: float) -> None
        Visualizes curvature and roughness with Open3D.
    pcd_snapshot_renderer(all_points: np.ndarray, all_curvatures: np.ndarray, all_roughness: np.ndarray, curvature_radius: float, roughness_radius: float, output_dir: Path) -> None
        Renders point cloud snapshots with curvature and roughness.
    export_results(all_points: np.ndarray, all_curvatures: np.ndarray, all_roughness: np.ndarray, output_dir: Path, filename: Path) -> None
        Exports computed curvature and roughness values by appending them to the original data
        and writing out a CSV-formatted text file.
    check_and_clean_for_nans(all_curvatures: np.ndarray) -> np.ndarray
        Validates and clears NaN values from curvature or roughness arrays.
    calculate_curvature_and_roughness(config: Dict[str, Any]) -> None
        Main pipeline for reading, processing, visualizing, and exporting point cloud data.
    cpu_memory_monitoring() -> Generator[List[int], None, None]
        Monitors CPU memory usage within a context manager.
    gpu_memory_monitoring() -> Generator[None, None, None]
        Monitors GPU memory usage within a context manager.
    main() -> None
        Orchestrates the entire process by loading config, running the pipeline, and measuring resource usage.

Contributor: fzhcis@rit.edu
Version: 1.0
Last Updated: 12/30/2024
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
from typing import List, Tuple, Dict, Any, Generator
from open3d.visualization import rendering


def load_config(json_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config

def pad_neighbors(points: torch.Tensor, 
                  neighbors_list: List[torch.Tensor], 
                  max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure that each point has the same number of 
    neighbors for batch processing.

    Args:
        points (torch.Tensor): Tensor of points.
        neighbors_list (List[torch.Tensor]): List of neighbor indices for each point.
        max_neighbors (int): Maximum number of neighbors to pad to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded neighbors and mask tensors.
    """
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


def batch_neighborhood_search(points: torch.Tensor, 
                              radius: float, 
                              max_neighbors: int, 
                              device: str) -> List[torch.Tensor]:
    """Perform neighborhood search on the GPU in batches.

    Args:
        points (torch.Tensor): Tensor of points.
        radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to consider.
        device (str): Device to perform the computation on.

    Returns:
        List[torch.Tensor]: List of neighbor indices for each point.
    """
    points = points.to(device)
    dist_matrix = torch.cdist(points, points)
    neighbors_list = []

    for i in range(len(points)):
        neighbors = (dist_matrix[i] <= radius).nonzero(as_tuple=True)[0]
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]
        neighbors_list.append(neighbors)
    return neighbors_list



def calculate_neighbor_eigens(points: np.ndarray, 
                              nn_radius: float, 
                              max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Calculate eigenvalues and eigenvectors for each point's neighborhood.

    Args:
        points (np.ndarray): Array of points.
        nn_radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to consider.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Eigenvalues, eigenvectors, and centered neighbors.
    """

    points = torch.tensor(points, dtype=torch.float32, device='cuda')
    neighbors_list = batch_neighborhood_search(points, nn_radius, max_neighbors, device='cuda')
    padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)
    
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    
    return eigenvalues, eigenvectors, (centered_neighbors, mask)

def estimate_curvature_roughness_batched(points: np.ndarray, 
                                         curvature_radius: float = 0.05, 
                                         roughness_radius: float = 0.2, 
                                         max_neighbors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate curvature and roughness for a batch of points with GPU-based neighborhood search.

    Args:
        points (np.ndarray): Array of points.
        curvature_radius (float, optional): Radius for curvature estimation. Defaults to 0.05.
        roughness_radius (float, optional): Radius for roughness estimation. Defaults to 0.2.
        max_neighbors (int, optional): Maximum number of neighbors to consider. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Curvature and roughness arrays.
    """

    # Curvature estimation
    curv_eigenval, _, _ = calculate_neighbor_eigens(points, curvature_radius, max_neighbors)
    curvatures = curv_eigenval[:, 0] / curv_eigenval.sum(dim=1)

    # Roughness estimation
    _, rough_eigenvec, neighbor_tuple = calculate_neighbor_eigens(points, roughness_radius, max_neighbors)
    centered_neighbors, mask = neighbor_tuple
    normal_vectors = rough_eigenvec[:, :, 0]
    roughness = torch.abs((centered_neighbors * mask.unsqueeze(2)).matmul(normal_vectors.unsqueeze(2)).squeeze()).mean(dim=1)

    return curvatures.cpu().numpy(), roughness.cpu().numpy()

def process_batches(dfs: List[pd.DataFrame], curvature_radius: float, roughness_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process point cloud batches to calculate curvature and roughness.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrame batches.
        curvature_radius (float): Radius for curvature estimation.
        roughness_radius (float): Radius for roughness estimation.
        max_neighbors (int): Maximum number of neighbors to consider.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Points, curvature, and roughness arrays.
    """
    all_points, all_curvatures, all_roughness = [], [], []
    for df_batch in tqdm(dfs, desc="Processing Batches", unit="batch"):
        points = df_batch[['X', 'Y', 'Z']].to_numpy()
        curvatures, roughness = estimate_curvature_roughness_batched(points, curvature_radius, roughness_radius, max_neighbors=max_neighbors)
        all_points.append(points)
        all_curvatures.append(curvatures)
        all_roughness.append(roughness)
    return np.vstack(all_points), np.hstack(all_curvatures), np.hstack(all_roughness)

def interactive_visualize_pcd(all_points: np.ndarray, 
                      all_curvatures: np.ndarray, 
                      all_roughness: np.ndarray,
                      curvature_radius: float,
                      roughness_radius: float) -> None:
    """Visualize curvature and roughness with Open3D.

    Args:
        all_points (np.ndarray): Array of points.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        curvature_radius (float): Radius for curvature estimation.
        roughness_radius (float): Radius for roughness estimation.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colormap = plt.get_cmap('plasma')
    all_curvatures_colors = colormap(all_curvatures)[:, :3]
    all_roughness_colors = colormap(all_roughness)[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(all_curvatures_colors)
    print("Displaying curvature visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Curvature Visualization (r={curvature_radius})")

    pcd.colors = o3d.utility.Vector3dVector(all_roughness_colors)
    print("Displaying roughness visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Roughness Visualization (r={roughness_radius})")

def pcd_snapshot_renderer(all_points: np.ndarray, 
                        all_curvatures: np.ndarray, 
                        all_roughness: np.ndarray,
                        curvature_radius: float,
                        roughness_radius: float,
                        output_dir: Path) -> None:
    """Render point cloud snapshots with curvature and roughness.

    Args:
        all_points (np.ndarray): Array of points.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        curvature_radius (float): Radius for curvature estimation.
        roughness_radius (float): Radius for roughness estimation.
        output_dir (Path): Output directory.

    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colormap = plt.get_cmap('plasma')
    
    renderer = rendering.OffscreenRenderer(400, 400)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # Set up the camera
    focus_point = [0, 0, 0]  
    camera_eye = [-2, -2, 1]  # Adjust for a better view
    camera_up = [0, 0, 1]
    renderer.scene.camera.look_at(focus_point, camera_eye, camera_up)

    # Curvature snapshot
    pcd.colors = o3d.utility.Vector3dVector(colormap(all_curvatures)[:, :3])
    renderer.scene.add_geometry("cloud", pcd, mat)
    snapshot = renderer.render_to_image()
    o3d.io.write_image(str(output_dir / f"snapshot_curvature_{curvature_radius}.png"), snapshot)

    # Roughness snapshot
    pcd.colors = o3d.utility.Vector3dVector(colormap(all_roughness)[:, :3])
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("cloud", pcd, mat)
    snapshot = renderer.render_to_image()
    o3d.io.write_image(str(output_dir / f"snapshot_roughness_{roughness_radius}.png"), snapshot)

    print("Snapshots saved to disk.")

    


def export_results(all_points: np.ndarray, 
                   all_curvatures: np.ndarray, 
                   all_roughness: np.ndarray, 
                   output_dir: Path, 
                   filename: Path) -> None:
    """Append curvature and roughness to points and export.

    Args:
        all_points (np.ndarray): Array of points.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        output_dir (Path): Output directory.
        filename (Path): Filename for the exported file.
    """
    df_filtered = pd.DataFrame(all_points, columns=['X', 'Y', 'Z'])
    df_filtered['curvature'] = all_curvatures
    df_filtered['roughness'] = all_roughness
    export_path = output_dir / f"{filename.stem}_curvature_roughness.txt"
    df_filtered.to_csv(export_path, sep=',', index=False)
    print(f"Exported point cloud with curvature and roughness to {export_path}")

def check_and_clean_for_nans(all_curvatures: np.ndarray) -> np.ndarray:
    """Check for NaN values in curvatures and roughness.

    Args:
        all_curvatures (np.ndarray): Array of curvature values.

    Returns:
        np.ndarray: Mask indicating valid values.
    """
    valid_mask = np.ones_like(all_curvatures, dtype=bool)
    if np.isnan(all_curvatures).any():
        print("Warning: NaN values detected in curvature/roughness calculation. Removing NaNs for histogram.")
        valid_mask = ~np.isnan(all_curvatures)
    
    if len(all_curvatures) == 0:
        raise ValueError("Error: No valid curvature or roughness values to plot.")
        
    return valid_mask

def calculate_curvature_and_roughness(config: Dict[str, Any]) -> None:
    """Process the point cloud, estimate curvature and roughness, and combine results.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    filename = Path(config["filename"])
    output_dir = Path(config["output_dir"])
    curvature_radius = config["curvature_radius"]
    roughness_radius = config["roughness_radius"]
    max_neighbors = config["max_neighbors"]
    batch_num = config["batch_num"]
    range1metres_min = config.get("range1metres_min", 2.25)
    range1metres_max = config.get("range1metres_max", 4.0)
    histogram_saveflag = config.get("histogram_saveflag", True)
    visualize = config.get("visualize", True)
    export = config.get("export", True)

    # df_filtered = preprocess_point_cloud(filename, range1metres_min, range1metres_max, clean_pc=False, upside_down=False)
    # Since filename is .ply, we can use o3d.io.read_point_cloud
    pt_xyz_np = o3d.io.read_point_cloud(str(filename)).points
    df_filtered = pd.DataFrame(pt_xyz_np, columns=['X', 'Y', 'Z']).reset_index(drop=True)

    print(f"Filtered point cloud shape: {df_filtered.shape}")

    num_points = df_filtered.shape[0]
    if num_points > 30_000:
        print("*********Large dataset detected. Processing in batches...*********")
        # df_grouped = df_filtered.groupby(pd.cut(df_filtered['azimuth'], batch_num))
        df_grouped = df_filtered.groupby(pd.cut(df_filtered['Z'], batch_num))
        dfs = [group for _, group in df_grouped]
        all_points, all_curvatures, all_roughness = process_batches(dfs, curvature_radius, roughness_radius, max_neighbors)
    else:
        print("*********Small dataset detected. Processing all at once...*********")
        all_points = df_filtered[['X', 'Y', 'Z']].to_numpy()
        all_curvatures, all_roughness = estimate_curvature_roughness_batched(all_points, curvature_radius, roughness_radius, max_neighbors)

    valid_mask_curv = check_and_clean_for_nans(all_curvatures)
    valid_mask_rough = check_and_clean_for_nans(all_roughness)
    valid_mask = valid_mask_curv & valid_mask_rough
    valid_mask = valid_mask_curv

    print(f"Number of valid points: {valid_mask.sum()}")
    all_points = all_points[valid_mask]
    df_filtered_out = df_filtered[valid_mask]
    all_curvatures = all_curvatures[valid_mask]
    all_roughness = all_roughness[valid_mask]



    normalized_curvatures = (all_curvatures - all_curvatures.min()) / (all_curvatures.max() - all_curvatures.min())
    normalized_roughness = (all_roughness - all_roughness.min()) / (all_roughness.max() - all_roughness.min())

    if histogram_saveflag:
        get_histogram(all_curvatures, output_dir, 
                    title="Curvature", 
                    saveflag=True, 
                    log_y=True)
        get_histogram(all_roughness, output_dir, 
                    title="Roughness", 
                    saveflag=True, 
                    log_y=True)
        get_histogram(normalized_curvatures, output_dir, 
                    title="Normalized Curvature", 
                    saveflag=True, 
                    log_y=True)
        get_histogram(normalized_roughness, output_dir, 
                    title="Normalized Roughness", 
                    saveflag=True, 
                    log_y=True)

    if visualize:
        interactive_visualize_pcd(all_points, normalized_curvatures, normalized_roughness, curvature_radius, roughness_radius)
        
    if export:    
        pcd_snapshot_renderer(all_points, 
                              normalized_curvatures, 
                              normalized_roughness, 
                              curvature_radius, 
                              roughness_radius, 
                              output_dir)
        export_results(all_points, 
                       normalized_curvatures, 
                       normalized_roughness, 
                       output_dir, 
                       filename)

@contextmanager
def cpu_memory_monitoring() -> Generator[List[int], None, None]:
    """Context manager for monitoring CPU memory usage.

    Yields:
        Generator[List[int], None, None]: Peak memory usage list.
    """
    process = psutil.Process()
    peak_memory = [0]
    stop_event = threading.Event()

    def monitor_memory():
        while not stop_event.is_set():
            mem = process.memory_info().rss
            if mem > peak_memory[0]:
                peak_memory[0] = mem
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    try:
        yield peak_memory
    finally:
        stop_event.set()
        monitor_thread.join()

@contextmanager
def gpu_memory_monitoring() -> Generator[None, None, None]:
    """Context manager for monitoring GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        pass

def main() -> None:
    """Main function to execute the point cloud processing script."""
    start_time = time.time()
    
    with cpu_memory_monitoring() as peak_memory:
        with gpu_memory_monitoring():
            config_path = Path('./input_params/calc_curvature_roughness_input_zmachine.json')
            config = load_config(config_path)
            calculate_curvature_and_roughness(config)
    
    elapsed_time = time.time() - start_time
    peak_cpu_memory_mb = peak_memory[0] / (1024 * 1024)

    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_memory_mb = 0

    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Peak CPU Memory Used: {peak_cpu_memory_mb:.2f} MB")
    print(f"Peak GPU Memory Used: {peak_gpu_memory_mb:.2f} MB")

if __name__ == "__main__":
    main()

