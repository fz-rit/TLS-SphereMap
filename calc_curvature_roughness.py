"""
calc_curvature_roughness.py
Contributor: fzhcis@rit.edu
Version: 2.1
Last Updated: 02/26/2024

This script computes curvature and roughness for 3D point clouds using GPU acceleration. 
It runs as a standalone module with a user-defined configuration file.
Key features:
- Curvature and roughness estimation for surface characterization.  
- GPU-accelerated computation and batch processing for large-scale datasets.  
- Display histograms of curvature and roughness values.
- Interactive visualization with Open3D, including normalized curvature and roughness maps.  
- Data export: saves computed values as .txt and snapshots of visualizations as .png.  
- Monitoring of CPU and GPU memory usage.  
"""
import open3d as o3d
import numpy as np
import pandas as pd
import time
import torch
from torch import Tensor
import psutil
from pathlib import Path
from tqdm import tqdm
from plot_tools import get_vector_histogram
from matplotlib import pyplot as plt
import threading
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Generator
from open3d.visualization import rendering
from config_loader import CONFIG # Configuration dictionary read from a .json file

def batch_neighborhood_search(points_xyz: Tensor, 
                              radius: float, 
                              max_neighbors: int, 
                              device: str) -> List[Tensor]:
    """Perform neighborhood search on the GPU in batches.

    Args:
        points_xyz (Tensor): Tensor of points_xyz.
        radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to consider.
        device (str): Device to perform the computation on.

    Returns:
        List[Tensor]: List of neighbor indices for each point.
    """
    points_xyz = points_xyz.to(device)
    dist_matrix = torch.cdist(points_xyz, points_xyz)
    neighbors_list = []

    for i in range(len(points_xyz)):
        neighbors = (dist_matrix[i] <= radius).nonzero(as_tuple=True)[0]
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]
        neighbors_list.append(neighbors)
    return neighbors_list


def pad_neighbors(points_xyz: Tensor, 
                  neighbors_list: List[Tensor], 
                  max_neighbors: int) -> Tuple[Tensor, Tensor]:
    """Ensure that each point has the same number of 
    neighbors for batch processing.

    Args:
        points_xyz (Tensor): Tensor of points_xyz.
        neighbors_list (List[Tensor]): List of neighbor indices for each point.
        max_neighbors (int): Maximum number of neighbors to pad to.

    Returns:
        Tuple[Tensor, Tensor]: Padded neighbors and mask tensors.
    """
    padded_neighbors = []
    mask = [] # Mask to keep track of valid neighbors
    
    for idx in neighbors_list:
        neighbors = points_xyz[idx]
        if len(neighbors) < max_neighbors:
            padded = torch.cat([neighbors, torch.zeros((max_neighbors - len(neighbors), 3), device=points_xyz.device)])
            mask.append(torch.cat([torch.ones(len(neighbors), device=points_xyz.device), torch.zeros(max_neighbors - len(neighbors), device=points_xyz.device)]))
        else:
            padded = neighbors[:max_neighbors]
            mask.append(torch.ones(max_neighbors, device=points_xyz.device))
        
        padded_neighbors.append(padded)
    
    return torch.stack(padded_neighbors), torch.stack(mask)


def calculate_neighbor_eigens(points_xyz: np.ndarray, 
                              nn_radius: float, 
                              max_neighbors: int) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    """Calculate eigenvalues and eigenvectors for each point's neighborhood.

    Args:
        points_xyz (np.ndarray): Array of points_xyz.
        nn_radius (float): Radius for neighborhood search.
        max_neighbors (int): Maximum number of neighbors to consider.

    Returns:
        Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]: Eigenvalues, eigenvectors, and centered neighbors.
    """

    points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device='cuda')
    neighbors_list = batch_neighborhood_search(points_xyz, nn_radius, max_neighbors, device='cuda')
    padded_neighbors, mask = pad_neighbors(points_xyz, neighbors_list, max_neighbors)
    
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
    
    return eigenvalues, eigenvectors, (centered_neighbors, mask)


def estimate_curvature_roughness_batched(points_xyz: np.ndarray, 
                                         neighbor_radius: float = 0.05, 
                                         max_neighbors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate curvature and roughness for a batch of points_xyz with GPU-based neighborhood search.

    Args:
        points_xyz (np.ndarray): Array of points_xyz.
        neighbor_radius (float, optional): Radius for curvature & roughness estimation. Defaults to 0.05.
        max_neighbors (int, optional): Maximum number of neighbors to consider. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Curvature and roughness arrays.
    """

    # Curvature estimation
    curv_eigenval, rough_eigenvec, neighbor_tuple = calculate_neighbor_eigens(points_xyz, neighbor_radius, max_neighbors)
    curvatures = curv_eigenval[:, 0] / curv_eigenval.sum(dim=1)

    # Roughness estimation
    centered_neighbors, mask = neighbor_tuple
    normal_vectors = rough_eigenvec[:, :, 0]
    roughness = torch.abs((centered_neighbors * mask.unsqueeze(2)).matmul(normal_vectors.unsqueeze(2)).squeeze()).mean(dim=1)

    return curvatures.cpu().numpy(), roughness.cpu().numpy()


def process_batches(dfs: List[pd.DataFrame], neighbor_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process point cloud batches to calculate curvature and roughness.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrame batches.
        neighbor_radius (float): Radius for curvature & roughness estimation.
        max_neighbors (int): Maximum number of neighbors to consider.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Points, curvature, and roughness arrays.
    """
    all_points_xyz, all_curvatures, all_roughness = [], [], []
    for df_batch in tqdm(dfs, desc="Processing Batches", unit="batch"):
        points_xyz = df_batch[['X', 'Y', 'Z']].to_numpy()
        curvatures, roughness = estimate_curvature_roughness_batched(points_xyz, neighbor_radius, max_neighbors=max_neighbors)
        all_points_xyz.append(points_xyz)
        all_curvatures.append(curvatures)
        all_roughness.append(roughness)
    
    all_points_xyz = np.vstack(all_points_xyz)
    all_curvatures = np.hstack(all_curvatures)
    all_roughness = np.hstack(all_roughness)
    return all_points_xyz, all_curvatures, all_roughness


def interactive_visualize_pcd(all_points_xyz: np.ndarray, 
                      all_curvatures: np.ndarray, 
                      all_roughness: np.ndarray,
                      neighbor_radius: float,
                      ) -> None:
    """Visualize curvature and roughness with Open3D.

    Args:
        all_points_xyz (np.ndarray): Array of points_xyz.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        neighbor_radius (float): Radius for curvature & roughness estimation.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_xyz)
    colormap = plt.get_cmap('plasma')
    all_curvatures_colors = colormap(all_curvatures)[:, :3]
    all_roughness_colors = colormap(all_roughness)[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(all_curvatures_colors)
    print("Displaying curvature visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Curvature Visualization (r={neighbor_radius})")

    pcd.colors = o3d.utility.Vector3dVector(all_roughness_colors)
    print("Displaying roughness visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Roughness Visualization (r={neighbor_radius})")


def pcd_snapshot_renderer(all_points_xyz: np.ndarray, 
                        all_curvatures: np.ndarray, 
                        all_roughness: np.ndarray,
                        neighbor_radius: float,
                        output_dir: Path) -> None:
    """Render point cloud snapshots with curvature and roughness.

    Args:
        all_points_xyz (np.ndarray): Array of points.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        neighbor_radius (float): Radius for curvature & roughness estimation.
        output_dir (Path): Output directory.

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_xyz)
    colormap = plt.get_cmap('plasma')
    
    renderer = rendering.OffscreenRenderer(400, 400)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # Set up the camera
    focus_point = [0, 0, 0]  
    camera_eye = [-2, -2, 1]
    camera_up = [0, 0, 1]
    renderer.scene.camera.look_at(focus_point, camera_eye, camera_up)

    # Curvature snapshot
    pcd.colors = o3d.utility.Vector3dVector(colormap(all_curvatures)[:, :3])
    renderer.scene.add_geometry("cloud", pcd, mat)
    snapshot = renderer.render_to_image()
    o3d.io.write_image(str(output_dir / f"snapshot_curvature_{neighbor_radius}.png"), snapshot)

    # Roughness snapshot
    pcd.colors = o3d.utility.Vector3dVector(colormap(all_roughness)[:, :3])
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("cloud", pcd, mat)
    snapshot = renderer.render_to_image()
    o3d.io.write_image(str(output_dir / f"snapshot_roughness_{neighbor_radius}.png"), snapshot)

    print("Snapshots saved to disk.")


def export_results(all_points_allinone: pd.DataFrame, 
                   neighbor_radius: float, 
                   output_dir: Path, 
                   input_path: Path) -> None:
    """Append curvature and roughness to points and export.

    Args:
        all_points_allinone (pd.DataFrame): Point cloud DataFrame with curvature and roughness.
        output_dir (Path): Output directory.
        input_path (Path): Filename for the exported file.
    """

    export_path = output_dir / f"{input_path.stem}_curvature_{neighbor_radius:.2f}_roughness_{neighbor_radius:.2f}.txt"
    all_points_allinone.to_csv(export_path, sep=',', index=False)
    print(f"Exported point cloud with curvature and roughness to {export_path}")


def check_and_clean_for_nans(all_curv_or_rough: np.ndarray) -> np.ndarray:
    """Check for NaN values in curvatures and roughness.

    Args:
        all_curv_or_rough (np.ndarray): Array of curvature values or roughness values.

    Returns:
        np.ndarray: Mask indicating valid values.
    """
    valid_mask = np.ones_like(all_curv_or_rough, dtype=bool)
    if np.isnan(all_curv_or_rough).any():
        print("Warning: NaN values detected in curvature/roughness calculation. Removing NaNs for histogram.")
        valid_mask = ~np.isnan(all_curv_or_rough)
    
    if len(all_curv_or_rough) == 0:
        raise ValueError("Error: No valid curvature or roughness values to plot.")
        
    return valid_mask


def calculate_curvature_and_roughness(config: Dict[str, Any]) -> None:
    """Process the point cloud, estimate curvature and roughness, and combine results.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    global_params = config["global"]
    params = config["calc_curvature_roughness"]
    output_dir = Path(global_params["output_dir"])
    input_file_stem = global_params['input_file_stem']
    input_path = Path(output_dir / f"{input_file_stem}_filtered_normaled.txt")
    neighbor_radius = params["neighbor_radius"] # Radius for neighborhood search for both curvature and roughness
    max_neighbors = params["max_neighbors"]
    batch_num_elevation = params.get("batch_num_elevation", 2)
    batch_num_azimuth = params.get("batch_num_azimuth", 2)
    histogram_saveflag = params.get("histogram_saveflag", True)
    visualize = params.get("interactive_visualize", True)
    export = params.get("export", True)
    delete_intermediate_file = params.get("delete_intermediate_file", False)

    points_df = pd.read_csv(input_path, sep=',')
    print(f"The point cloud dataframe shape: {points_df.shape}")

    num_points = points_df.shape[0]
    if num_points > 30_000:
        print("*********Large dataset detected. Processing in batches...*********")
        df_grouped = points_df.groupby([pd.cut(points_df['azimuth'], batch_num_azimuth), 
                                        pd.cut(points_df['elevation'], batch_num_elevation)], 
                                        observed=False,
                                        sort=False)
        dfs = [group for _, group in df_grouped]
        all_points_allinone = pd.concat(dfs, ignore_index=True)  # Ignore the index when concatenating
        all_points_xyz, all_curvatures, all_roughness = process_batches(dfs, neighbor_radius, max_neighbors)
    else:
        print("*********Small dataset detected. Processing all at once...*********")
        all_points_allinone = points_df.copy()
        all_points_xyz = points_df[['X', 'Y', 'Z']].to_numpy()
        all_curvatures, all_roughness = estimate_curvature_roughness_batched(all_points_xyz, neighbor_radius, max_neighbors)

    ## Post-processing ##
    # Check for NaN values in curvatures and roughness
    valid_mask_curv = check_and_clean_for_nans(all_curvatures)
    valid_mask_rough = check_and_clean_for_nans(all_roughness)

    min_curvature = all_curvatures[valid_mask_curv].min() + 1e-4
    min_roughness = all_roughness[valid_mask_rough].min()

    # Assign min values (instead of 0) to invalid points, so that they are not lost in visualization
    all_curvatures[~valid_mask_curv] = max(min_curvature, 0) 
    all_roughness[~valid_mask_rough] = max(min_roughness, 0)
    all_curvatures[all_curvatures<0] = min_curvature

    # Attach the curvature and roughness values to the original point cloud dataframe.
    all_points_allinone['curvature'] = all_curvatures
    all_points_allinone['roughness'] = all_roughness
    if histogram_saveflag: # Save histograms of curvature and roughness
        get_vector_histogram(all_curvatures, output_dir, 
                    title="Curvature", 
                    saveflag=True, 
                    log_y=True)
        get_vector_histogram(all_roughness, output_dir, 
                    title="Roughness", 
                    saveflag=True, 
                    log_y=True)

    if visualize: # Interactive visualization with Open3D
        normalized_curvatures = (all_curvatures - all_curvatures.min()) / (all_curvatures.max() - all_curvatures.min())
        normalized_roughness = (all_roughness - all_roughness.min()) / (all_roughness.max() - all_roughness.min())
        interactive_visualize_pcd(all_points_xyz, normalized_curvatures, normalized_roughness, neighbor_radius)
        
    if export: # Export results to disk
        pcd_snapshot_renderer(all_points_xyz, 
                              all_curvatures, 
                              all_roughness, 
                              neighbor_radius, 
                              output_dir)
        export_results(all_points_allinone, 
                       neighbor_radius, 
                       output_dir, 
                       input_path)
    
    if delete_intermediate_file:
        input_path.unlink()
        print(f"Deleted intermediate file: {input_path}")

# Monitoring CPU and GPU memory usage
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
            time.sleep(0.1) # Check every 0.1 seconds

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
    start_time = time.time()
    
    with cpu_memory_monitoring() as peak_memory:
        with gpu_memory_monitoring():
            print("-----------Calculating curvature and roughness...----------")
            print("Configuration:")
            print(CONFIG['global'])
            print(CONFIG['calc_curvature_roughness'])

            calculate_curvature_and_roughness(CONFIG)
    
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

