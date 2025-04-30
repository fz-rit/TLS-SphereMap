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
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from tools.plot_tools import get_vector_histogram
from typing import List, Tuple, Dict, Any
from tools.pcd_utils import check_and_clean_for_nans, interactive_visualize_pcd, export_results, MemoryProfiler
from tools.config_loader import CONFIG # Configuration dictionary read from a .json file
from tools.pcd_utils import create_dir_if_not_exists

class CalcCurvatureRoughness:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.points_xyz = None

    def set_points(self, points_xyz: np.ndarray):
        """Set the points_xyz array as a tensor on the GPU."""
        self.points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device=self.device)

    def batch_neighborhood_search(self, radius: float, max_neighbors: int) -> List[Tensor]:
        """Perform neighborhood search on the GPU in batches.

        Args:
            radius (float): Radius for neighborhood search.
            max_neighbors (int): Maximum number of neighbors to consider.

        Returns:
            List[Tensor]: List of neighbor indices for each point.
        """
        dist_matrix = torch.cdist(self.points_xyz, self.points_xyz)
        neighbors_list = []

        for i in range(len(self.points_xyz)):
            neighbors = (dist_matrix[i] <= radius).nonzero(as_tuple=True)[0]
            if len(neighbors) > max_neighbors:
                neighbors = neighbors[:max_neighbors]
            neighbors_list.append(neighbors)
        return neighbors_list

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
        neighbors_list = self.batch_neighborhood_search(nn_radius, max_neighbors)
        padded_neighbors, mask = self.pad_neighbors(neighbors_list, max_neighbors)
        
        centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
        covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        
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


def process_point_cloud_curvature_roughness(config: Dict[str, Any]) -> pd.DataFrame:
    """Process the point cloud, estimate curvature and roughness, and combine results.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Load configuration parameters
    global_params = config["global"]
    params = config["calc_curvature_roughness"]
    output_dir = Path(global_params["output_dir"])
    input_file_stem = global_params['input_file_stem']
    input_path = Path(output_dir / 'pcd' / f"{input_file_stem}_filtered_normaled.txt")
    neighbor_radius = params["neighbor_radius"] # Radius for neighborhood search for both curvature and roughness
    max_neighbors = params["max_neighbors"]
    batch_num_elevation = params.get("batch_num_elevation", 2)
    batch_num_azimuth = params.get("batch_num_azimuth", 2)
    histogram_saveflag = params.get("histogram_saveflag", True)
    visualize = params.get("interactive_visualize", True)
    export = params.get("export", True)
    delete_intermediate_file = params.get("delete_intermediate_file", False)

    # Load the point cloud dataframe
    points_df = pd.read_csv(input_path, sep=',')
    print(f"The point cloud dataframe shape: {points_df.shape}")

    # Calculate curvature and roughness
    num_points = points_df.shape[0]
    calc_curv_rough = CalcCurvatureRoughness(device='cuda')
    if num_points > 30_000:
        print("*********Large dataset detected. Processing in batches...*********")
        df_grouped = points_df.groupby([pd.cut(points_df['azimuth'], batch_num_azimuth), 
                                        pd.cut(points_df['elevation'], batch_num_elevation)], 
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
            print("-----------Calculating curvature and roughness...----------")
            print("Configuration:")
            print(CONFIG['global'])
            print(CONFIG['calc_curvature_roughness'])

            process_point_cloud_curvature_roughness(CONFIG)
    
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