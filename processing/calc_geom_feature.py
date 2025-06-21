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
from tools.preprocess_point_cloud import read_and_clean_pcd
from tools.pcd_utils import check_and_clean_for_nans, interactive_visualize_pcd, export_results, MemoryProfiler
from tools.config_loader import CONFIG # Configuration dictionary read from a .json file
from tools.pcd_utils import create_dir_if_not_exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


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


class GeometricFeatureCalculator:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.points_xyz = None

    def set_points_on_gpu(self, points_xyz: np.ndarray):
        """Set the points_xyz array as a tensor on the GPU."""
        self.points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device=self.device)

    def adaptive_radius(self, 
                    base_radius=0.05, 
                    min_pts=20, 
                    scale=1.5, 
                    adaptive_r_max=0.2, 
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
            # plt.show()

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

    def estimate_geometric_features_batched(
                                            self,
                                            neighbor_radius: float = 0.05,
                                            max_neighbors: int = 10
                                        ) -> Dict[str, np.ndarray]:
        """
        Estimate curvature, roughness, anisotropy, and surface variation for a batch of points.

        Args:
            neighbor_radius (float, optional): Radius for neighborhood search. Defaults to 0.05.
            max_neighbors (int, optional): Max number of neighbors to consider. Defaults to 10.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the estimated geometric features.
        """
        eigenvals, eigenvecs, (centered_neighbors, mask) = self.calculate_neighbor_eigens(
            neighbor_radius, max_neighbors
        )

        # Sort eigenvalues in ascending order: lambda1 ≤ lambda2 ≤ lambda3
        lambda1 = eigenvals[:, 0]
        lambda2 = eigenvals[:, 1]
        lambda3 = eigenvals[:, 2]
        lambda_sum = eigenvals.sum(dim=1)

        # -- Feature 1: Curvature
        curvature = lambda1 / lambda_sum

        # # -- Feature 2: Roughness (transformed)
        # normal_vectors = eigenvecs[:, :, 0]  # smallest eigenvalue's eigenvector
        # dot_prods = (centered_neighbors * mask.unsqueeze(2)).matmul(normal_vectors.unsqueeze(2)).squeeze()
        # roughness = torch.abs(dot_prods).mean(dim=1)
        # roughness_trans = torch.sqrt(roughness)  # optional transformation for visualization

        # -- Feature 3: Anisotropy
        anisotropy = (lambda3 - lambda2) / lambda3.clamp(min=1e-9)

        # # -- Feature 4: Surface Variation
        # surface_variation = lambda3 / lambda_sum

        # -- Feature 5: Planarity
        planarity = (lambda2 - lambda1) / lambda3.clamp(min=1e-9)

        # -- Normals at x, y, z directions
        # Normal vectors (smallest eigenvector direction)
        normals = eigenvecs[:, :, 0]  # Shape: [N, 3]

        # Ensure normals point toward origin (0, 0, 0)
        to_origin = -self.points_xyz  # Shape: [N, 3]
        flip_mask = (normals * to_origin).sum(dim=1) < 0
        normals[flip_mask] *= -1
        normals = normals.cpu().numpy()
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

        out_feat_dict = {
            'curvature': curvature.cpu().numpy(),
            # 'roughness': roughness.cpu().numpy(),
            'anisotropy': anisotropy.cpu().numpy(),
            # 'surface_variation': surface_variation.cpu().numpy(),
            'planarity': planarity.cpu().numpy(),
            'nx': nx,
            'ny': ny,
            'nz': nz
        }
        return out_feat_dict



    def process_batches_with_buffer(self, 
                                dfs: List[pd.DataFrame], 
                                all_df: pd.DataFrame, 
                                buffer_pts: int,
                                neighbor_radius: float, 
                                max_neighbors: int
                               ) -> pd.DataFrame:
        """
        Process spatial batches with buffered neighborhoods using pandas DataFrames.

        Returns a single DataFrame with geometric features added to the core points.
        """
        from sklearn.neighbors import KDTree

        def extract_buffered_batch_by_knn(core_df, all_df, buffer_k=30):
            all_points = all_df[['X', 'Y', 'Z']].to_numpy()
            core_points = core_df[['X', 'Y', 'Z']].to_numpy()

            kdtree = KDTree(all_points)
            indices = kdtree.query(core_points, k=buffer_k, return_distance=False)
            neighbor_indices = set(indices.flatten())

            buffered_df = all_df.iloc[list(neighbor_indices)].copy()
            core_mask = buffered_df.index.isin(core_df.index)
            return buffered_df, core_mask

        result_dfs = []

        for df_batch in tqdm(dfs, desc="Processing Batches with Buffer", unit="batch"):
            buffered_df, core_mask = extract_buffered_batch_by_knn(df_batch, all_df, buffer_pts)
            points_xyz = buffered_df[['X', 'Y', 'Z']].to_numpy()
            self.set_points_on_gpu(points_xyz)

            geom_feat_dict = self.estimate_geometric_features_batched(
                neighbor_radius, max_neighbors
            )

            core_df = buffered_df.loc[core_mask].copy()
            for feat_name, feat_values in geom_feat_dict.items():
                if feat_name in core_df.columns:
                    print(f"Warning: {feat_name} already exists in core_df. Overwriting.")
                    print(f"Before overwriting, range of {feat_name}: {core_df[feat_name].min()} to {core_df[feat_name].max()}")
                core_df[feat_name] = feat_values[core_mask]

            result_dfs.append(core_df)

        geom_feat_names = ['curvature', 'anisotropy', 'planarity']
        return pd.concat(result_dfs), geom_feat_names



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

def generate_geom_feat_from_pcd(params: Dict[str, Any], points_df, output_dir) -> pd.DataFrame:
    """Process the point cloud, estimate curvature and roughness, and combine results."""
    input_file_stem = output_dir.parent.name

    # Load config
    neighbor_radius = params["base_radius"]
    out_signature_str = params.get("out_signature_str", "geom_feat")
    buffer_pts = params.get("buffer_pts", 30)  # Default to 30 neighbors for buffer
    max_neighbors = params["max_neighbors"]
    pts_num_per_batch = params.get("pts_num_per_batch", 10_000)
    histogram_saveflag = params.get("histogram_saveflag", True)
    visualize = params.get("interactive_visualize", True)
    export = params.get("export", True)

    num_points = points_df.shape[0]
    geom_feature_calculator = GeometricFeatureCalculator(device='cuda')

    if num_points > 30_000:
        print("Large dataset detected. Processing in batches...")
        batch_num_x, batch_num_y = infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch)
        df_grouped = points_df.groupby([
            pd.cut(points_df['azimuth'], batch_num_x), 
            pd.cut(points_df['elevation'], batch_num_y)
        ], observed=False, sort=False)
        dfs = [group for _, group in df_grouped]
        result_df, geom_feat_names = geom_feature_calculator.process_batches_with_buffer(
            dfs=dfs,
            all_df=points_df,
            buffer_pts=buffer_pts,
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors
        )
    else:
        print("Small dataset. Processing all at once...")
        result_df = points_df.copy()
        all_xyz = result_df[['X', 'Y', 'Z']].to_numpy()
        geom_feat_dict = geom_feature_calculator.estimate_geometric_features_batched(all_xyz, 
                                                                  neighbor_radius=neighbor_radius, 
                                                                  max_neighbors=max_neighbors)
        geom_feat_names = list(geom_feat_dict.keys())
        for feat_name, feat_values in geom_feat_dict.items():
            result_df[feat_name] = feat_values

    # Clean-up NaNs and assign defaults
    for feat_name in geom_feat_names:
        valid = result_df[feat_name].notna()
        min_val = result_df.loc[valid, feat_name].min() + (1e-4 if feat_name == 'curvature' else 0)
        result_df.loc[~valid, feat_name] = max(min_val, 0)
        if feat_name == 'curvature':
            result_df.loc[result_df[feat_name] < 0, feat_name] = min_val

    if histogram_saveflag:
        for feat_name in geom_feat_names:
            if feat_name in result_df.columns:
                print(f"Generating histogram for {feat_name}...")
                get_vector_histogram(result_df[feat_name].values, output_dir, title=feat_name)

    if visualize:
        print("Normalizing geometric features for visualization...")
        all_geom_features = []
        for feat_name in geom_feat_names:
            normalized_feat = (result_df[feat_name] - result_df[feat_name].min()) / (result_df[feat_name].max() - result_df[feat_name].min())
            normalized_feat.to_numpy()
            all_geom_features.append(normalized_feat)

        all_points_xyz = result_df[['X', 'Y', 'Z']].to_numpy()
        if len(all_geom_features) != len(geom_feat_names):
            raise ValueError("Mismatch between number of geometric features and their names.")
        print("Visualizing point cloud with geometric features...")
        interactive_visualize_pcd(all_points_xyz, 
                                  all_geom_features=all_geom_features, 
                                  geom_feature_names=geom_feat_names)

    if export:
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_results(result_df, out_signature_str, save_dir, input_file_stem)


    return result_df



def main() -> None:
    start_time = time.time()
    profiler = MemoryProfiler()
    with profiler.cpu_memory_monitoring() as peak_memory:
        with profiler.gpu_memory_monitoring():
            print("-----------Calculating geometric features such as curvature, anisotropy, and planarity.----------")
            print("Configuration:")
            global_params = CONFIG["global"]
            current_file_params = CONFIG["calc_geom_feature"]
            output_dir_ls = global_params["output_dir_ls"]
            input_path_ls = global_params["input_path_ls"]
            cut_percent = current_file_params["cut_percent"]
            clean_pc = current_file_params["clean_pc"]
            flip_mangrove = current_file_params["flip_mangrove"]
            dataset_name = global_params["dataset"]
            # for output_dir in output_dir_ls:
            for input_path, output_dir in zip(input_path_ls, output_dir_ls):
                print(f'#######Processing {input_path.stem}...########')
                points_df = read_and_clean_pcd(input_path, 
                                                cut_percent=cut_percent,
                                                clean_pc=clean_pc,
                                                dataset_name=dataset_name,
                                                flip_mangrove=flip_mangrove)
                generate_geom_feat_from_pcd(current_file_params, points_df, output_dir)

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