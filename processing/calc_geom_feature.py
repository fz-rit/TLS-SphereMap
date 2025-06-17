"""
Contributor: fzhcis@rit.edu
Version: 2.1
Last Updated: 02/27/2024

Description:
This script calculates the curvature and roughness of a point cloud using the eigenvalues 
and eigenvectors of the covariance matrix of the neighborhood of each point.
The curvature is calculated as the ratio of the smallest eigenvalue to the sum of all eigenvalues.
The roughness is calculated as the absolute dot product of the centered neighbors with the normal vector of the neighborhood.
The script uses PyTorch for GPU-accelerated computation.

Usage:
1. Update the configuration file (e.g. 3D_to_2D_config_harvard_forest.json) with the desired paths and parameters.
2. Double-check the config_loader.py file to ensure that the correct configuration file is being loaded.
3. Run the script with the following command:
   python calc_curvature_roughness.py
"""
import numpy as np
import pandas as pd
import time
import torch
import random
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from tools.plot_tools import get_vector_histogram
from typing import List, Tuple, Dict, Any
from tools.pcd_utils import check_and_clean_for_nans, interactive_visualize_pcd, export_results, MemoryProfiler
from tools.config_loader import CONFIG # Configuration dictionary read from a .json file
from tools.pcd_utils import create_dir_if_not_exists
import matplotlib.pyplot as plt


import gc

def clear_memory(*vars_to_delete):
    """
    Frees up GPU and CPU memory between batches.

    Parameters
    ----------
    *vars_to_delete : list of variables
        Any large tensors or objects you want to explicitly delete.

    Example
    -------
    clear_memory(dist_matrix, neighbors_list)
    """
    for var in vars_to_delete:
        try:
            del var
        except Exception as e:
            print(f"[Warning] Could not delete variable: {e}")
    
    gc.collect()  # Force Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear unused GPU memory


class CalcCurvatureRoughness:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.points_xyz = None

    def set_points(self, points_xyz: np.ndarray):
        """Set the points_xyz array as a tensor on the GPU."""
        self.points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device=self.device)

    # def batch_neighborhood_search(self, radius: float, max_neighbors: int) -> List[Tensor]:
    #     """Perform neighborhood search on the GPU in batches.

    #     Args:
    #         radius (float): Radius for neighborhood search.
    #         max_neighbors (int): Maximum number of neighbors to consider.

    #     Returns:
    #         List[Tensor]: List of neighbor indices for each point.
    #     """
    #     dist_matrix = torch.cdist(self.points_xyz, self.points_xyz)
    #     neighbors_list = []

    #     for i in range(len(self.points_xyz)):
    #         neighbors = (dist_matrix[i] <= radius).nonzero(as_tuple=True)[0]
    #         if len(neighbors) > max_neighbors:
    #             neighbors = neighbors[:max_neighbors]
    #         neighbors_list.append(neighbors)
    #     return neighbors_list
    

    def adaptive_radius(self, 
                    base_radius=0.05, 
                    min_pts=20, 
                    scale=1.5, 
                    adaptive_r_max=0.3, 
                    debug=True, 
                    n_debug_samples=5,
                    visualize=True):
        """
        Estimate adaptive neighborhood radius per point based on local density.

        Parameters:
        -----------
        base_radius : float
            Minimum allowable radius (fallback for sparse regions).
        min_pts : int
            Minimum number of neighbors to define local density.
        scale : float
            Multiplier for the adaptive radius.
        adaptive_r_max : float
            Maximum allowable adaptive radius.
        debug : bool
            If True, logs radius info for a few random points.
        n_debug_samples : int
            Number of points to log in debug mode (default 5).
        visualize : bool
            If True, plots a histogram of adaptive radii (log scale).

        Returns:
        --------
        neighbors_list : List[Tensor]
            List of neighbor indices for each point.
        adaptive_radii : List[float]
            List of adaptive radii used for each point.
        """
        dist_matrix = torch.cdist(self.points_xyz, self.points_xyz)
        neighbors_list = []
        adaptive_radii = []

        num_points = self.points_xyz.shape[0]
        sample_indices = random.sample(range(num_points), min(n_debug_samples, num_points)) if debug else []

        for i in range(num_points):
            dists = dist_matrix[i]
            sorted_dists, _ = torch.sort(dists)

            # Compute adaptive radius and clamp
            raw_adaptive_r = sorted_dists[min_pts] * scale
            adaptive_r = torch.clamp(raw_adaptive_r, min=base_radius, max=adaptive_r_max)

            neighbors = (dists <= adaptive_r).nonzero(as_tuple=True)[0]
            neighbors_list.append(neighbors)
            adaptive_radii.append(adaptive_r.item())

            if i in sample_indices:
                print(f"[Debug] Point {i}: raw={raw_adaptive_r.item():.4f}, clamped={adaptive_r.item():.4f}, neighbors={len(neighbors)}")

        if visualize:
            plt.figure(figsize=(6, 4))
            plt.hist(adaptive_radii, bins=50, color='skyblue', edgecolor='gray')
            plt.yscale('log')
            plt.xlabel("Clamped Adaptive Radius (m)")
            plt.ylabel("Log-scaled Count")
            plt.title("Distribution of Adaptive Radii (Log Y-scale)")
            plt.tight_layout()
            plt.show()

        return neighbors_list, adaptive_radii

    def pad_neighbors(self, neighbors_list: List[Tensor], max_neighbors: int) -> Tuple[Tensor, Tensor]:
        """Ensure that each point has the same number of 
        neighbors for batch processing.

        Args:
            neighbors_list (List[Tensor]): List of neighbor indices for each point.
            max_neighbors (int): Maximum number of neighbors to pad to.

        Returns:
            Tuple[Tensor, Tensor]: Padded neighbors and mask tensors.
        """
        padded_neighbors, mask = [], [] # Use mask to keep track of valid neighbors
        
        for idx in neighbors_list:
            neighbors = self.points_xyz[idx]
            if len(neighbors) < max_neighbors:
                padded = torch.cat([neighbors, torch.zeros((max_neighbors - len(neighbors), 3), device=self.device)])
                mask.append(torch.cat([torch.ones(len(neighbors), device=self.device), torch.zeros(max_neighbors - len(neighbors), device=self.device)]))
            else:
                padded = neighbors[:max_neighbors]
                mask.append(torch.ones(max_neighbors, device=self.device))
            
            padded_neighbors.append(padded)
        
        return torch.stack(padded_neighbors), torch.stack(mask)

    def calculate_neighbor_eigens(self, nn_radius: float, max_neighbors: int) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Calculate eigenvalues and eigenvectors for each point's neighborhood.

        Args:
            nn_radius (float): Radius for neighborhood search.
            max_neighbors (int): Maximum number of neighbors to consider.

        Returns:
            Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]: Eigenvalues, eigenvectors, and centered neighbors.
        """
        # neighbors_list = self.batch_neighborhood_search(nn_radius, max_neighbors)
        # Use adaptive radius for neighborhood search
        neighbors_list, _ = self.adaptive_radius(base_radius=nn_radius, visualize=False, debug=False, n_debug_samples=0)
        padded_neighbors, mask = self.pad_neighbors(neighbors_list, max_neighbors)
        
        
        centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
        covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        
        clear_memory(neighbors_list, padded_neighbors, centroids, covariances)
        return eigenvalues, eigenvectors, (centered_neighbors, mask)

    def estimate_curvature_roughness_batched(self, neighbor_radius: float = 0.05, max_neighbors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate curvature and roughness for a batch of points_xyz with GPU-based neighborhood search.

        Args:
            neighbor_radius (float, optional): Radius for curvature & roughness estimation. Defaults to 0.05.
            max_neighbors (int, optional): Maximum number of neighbors to consider. Defaults to 10.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Curvature and roughness arrays.
        """
        # Curvature estimation
        curv_eigenval, rough_eigenvec, neighbor_tuple = self.calculate_neighbor_eigens(neighbor_radius, max_neighbors)
        curvatures = curv_eigenval[:, 0] / curv_eigenval.sum(dim=1)

        # Roughness estimation
        centered_neighbors, mask = neighbor_tuple
        normal_vectors = rough_eigenvec[:, :, 0]
        roughness = torch.abs((centered_neighbors * mask.unsqueeze(2)).matmul(normal_vectors.unsqueeze(2)).squeeze()).mean(dim=1)

        # Planularity estimation (optional, can be added if needed)
        # planarity = (curv_eigenval[:, 1] - curv_eigenval[:, 2]) / curv_eigenval[:, 0]

        

        return curvatures.cpu().numpy(), roughness.cpu().numpy()

    def process_batches(self, dfs: List[pd.DataFrame], neighbor_radius: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process point cloud batches to calculate curvature and roughness.

        Args:
            dfs (List[pd.DataFrame]): A list of DataFrame batches.
            neighbor_radius (float): Radius for curvature & roughness estimation.
            max_neighbors (int): Maximum number of neighbors to consider.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Points, curvature, and roughness arrays.
        """
        all_points_xyz, all_curvatures, all_roughness = [], [], []
        for df_batch in tqdm(dfs, desc="Processing Batches", unit="batch"):
            points_xyz = df_batch[['X', 'Y', 'Z']].to_numpy()
            self.set_points(points_xyz)
            curvatures, roughness = self.estimate_curvature_roughness_batched(neighbor_radius, max_neighbors=max_neighbors)
            all_points_xyz.append(points_xyz)
            all_curvatures.append(curvatures)
            all_roughness.append(roughness)
        
        all_points_xyz = np.vstack(all_points_xyz)
        all_curvatures = np.hstack(all_curvatures)
        all_roughness = np.hstack(all_roughness)
        return all_points_xyz, all_curvatures, all_roughness


def infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch: int = 10_000) -> Tuple[int, int]:
    """Infer the number of azimuth and elevation batches based on the point cloud size, azimuth, and elevation ranges.
    First, calculate the range of azimuth and elevation angles in the point cloud.
    Second, calculate the number of batches needed for azimuth and elevation based on the ratio of their ranges.

    Args:
        points_df (pd.DataFrame): DataFrame containing point cloud data.
        pts_num_per_batch (int, optional): Number of points per batch. Defaults to 10_000.

    Returns:
        Tuple[int, int]: Number of azimuth (x) and elevation batches (y).
    """
    azimuth_range = points_df['azimuth'].max() - points_df['azimuth'].min()
    elevation_range = points_df['elevation'].max() - points_df['elevation'].min()
    
    total_points = points_df.shape[0]
    x2y_ratio = azimuth_range / elevation_range if elevation_range != 0 else 1.0
    total_batches = int(np.ceil(total_points / pts_num_per_batch))
    # Calculate the number of batches for azimuth and elevation based on the ratio
    batch_num_x = int(np.ceil(np.sqrt(total_batches * x2y_ratio)))
    batch_num_y = int(np.ceil(total_batches / batch_num_x))

    print(f"Total points: {total_points}, Azimuth range: {azimuth_range}, Elevation range: {elevation_range}")
    print(f"Total batches: {batch_num_x*batch_num_y}, Batch num azimuth: {batch_num_x}, Batch num elevation: {batch_num_y}")

    return batch_num_x, batch_num_y


def generate_geom_feat_from_pcd(params: Dict[str, Any], output_dir) -> pd.DataFrame:
    """Process the point cloud, estimate curvature and roughness, and combine results.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Load configuration parameters
    input_file_stem = output_dir.parent.name
    input_path = Path(output_dir / 'pcd' / f"{input_file_stem}_filtered_normaled.txt")
    neighbor_radius = params["neighbor_radius"] # Radius for neighborhood search for both curvature and roughness
    max_neighbors = params["max_neighbors"]
    pts_num_per_batch = params.get("pts_num_per_batch", 10_000)
    histogram_saveflag = params.get("histogram_saveflag", True)
    visualize = params.get("interactive_visualize", True)
    export = params.get("export", True)
    delete_intermediate_file = params.get("delete_intermediate_file", False)

    points_df = pd.read_csv(input_path, sep=',')
    print(f"The point cloud dataframe shape: {points_df.shape}")
    batch_num_x, batch_num_y = infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch)

    # Calculate curvature and roughness
    num_points = points_df.shape[0]
    calc_curv_rough = CalcCurvatureRoughness(device='cuda')
    if num_points > 30_000:
        print("*********Large dataset detected. Processing in batches...*********")
        df_grouped = points_df.groupby([pd.cut(points_df['azimuth'], batch_num_x), 
                                        pd.cut(points_df['elevation'], batch_num_y)], 
                                        observed=False,
                                        sort=False)
        dfs = [group for _, group in df_grouped]
        all_points_allinone = pd.concat(dfs, ignore_index=True)  # Ignore the index when concatenating
        all_points_xyz, all_curvatures, all_roughness = calc_curv_rough.process_batches(dfs, neighbor_radius, max_neighbors)
    else:
        print("*********Small dataset detected. Processing all at once...*********")
        all_points_allinone = points_df.copy()
        all_points_xyz = points_df[['X', 'Y', 'Z']].to_numpy()
        all_curvatures, all_roughness = calc_curv_rough.estimate_curvature_roughness_batched(all_points_xyz, neighbor_radius, max_neighbors)

    # Post-processing
    valid_mask_curv = check_and_clean_for_nans(all_curvatures)
    valid_mask_rough = check_and_clean_for_nans(all_roughness)

    min_curvature = all_curvatures[valid_mask_curv].min() + 1e-4
    min_roughness = all_roughness[valid_mask_rough].min()

    all_curvatures[~valid_mask_curv] = max(min_curvature, 0) 
    all_roughness[~valid_mask_rough] = max(min_roughness, 0)
    all_curvatures[all_curvatures<0] = min_curvature # Assign min values (instead of 0) to invalid points, so that they are not lost in visualization

    all_points_allinone['curvature'] = all_curvatures
    all_points_allinone['roughness'] = all_roughness

    # Optional steps: Make histograms, Visualization, Export, and Delete intermediate file.
    if histogram_saveflag: # Display and save histograms of curvature and roughness
        get_vector_histogram(all_curvatures, output_dir, title="Curvature")
        get_vector_histogram(all_roughness, output_dir, title="Roughness")

    if visualize: # Interactive visualization with Open3D
        normalized_curvatures = (all_curvatures - all_curvatures.min()) / (all_curvatures.max() - all_curvatures.min())
        normalized_roughness = (all_roughness - all_roughness.min()) / (all_roughness.max() - all_roughness.min())
        interactive_visualize_pcd(all_points_xyz, normalized_curvatures, normalized_roughness, neighbor_radius)
        
    if export: # Export results to disk
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_results(all_points_allinone, 
                       neighbor_radius, 
                       save_dir, 
                       input_file_stem)
    
    if delete_intermediate_file:
        input_path.unlink()
        print(f"Deleted intermediate file: {input_path}")

    return all_points_allinone


def main() -> None:
    start_time = time.time()
    profiler = MemoryProfiler()
    with profiler.cpu_memory_monitoring() as peak_memory:
        with profiler.gpu_memory_monitoring():
            print("-----------Calculating geometric features such as curvature, roughness...----------")
            print("Configuration:")
            global_params = CONFIG["global"]
            current_file_params = CONFIG["calc_geom_feature"]
            output_dir_ls = global_params["output_dir_ls"]
            for output_dir in output_dir_ls:
                input_path_stem = output_dir.parent.name 
                print(f'#######Processing {input_path_stem}...########')
                generate_geom_feat_from_pcd(current_file_params, output_dir)
    
    # Monitor memory usage and elapsed time
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