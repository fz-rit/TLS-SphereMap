"""Geometric feature calculation for point clouds using PyTorch GPU acceleration."""
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
from tools.pcd_utils import interactive_visualize_pcd, export_results, MemoryProfiler
from tools.config_loader import get_config
from tools.pcd_utils import create_dir_if_not_exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import gc

def clear_memory(*vars_to_delete):
    for var in vars_to_delete:
        try:
            del var
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GeometricFeatureCalculator:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.points_xyz = None

    def set_points_on_gpu(self, points_xyz: np.ndarray):
        """Set the points_xyz array as a tensor on the GPU."""
        self.points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device=self.device)

    def adaptive_radius(self, base_radius=0.05, min_pts=20, scale=1.5, adaptive_r_max=0.2, debug=False, n_debug_samples=5, visualize=False):
        dist_matrix = torch.cdist(self.points_xyz, self.points_xyz)
        neighbors_list = []
        adaptive_radii = []

        num_points = self.points_xyz.shape[0]
        sample_indices = random.sample(range(num_points), min(n_debug_samples, num_points)) if debug else []

        for i in range(num_points):
            dists = dist_matrix[i]
            sorted_dists, _ = torch.sort(dists)
            raw_adaptive_r = sorted_dists[min_pts] * scale
            adaptive_r = torch.clamp(raw_adaptive_r, min=base_radius, max=adaptive_r_max)
            neighbors = (dists <= adaptive_r).nonzero(as_tuple=True)[0]
            neighbors_list.append(neighbors)
            adaptive_radii.append(adaptive_r.item())

            if i in sample_indices:
                print(f"Point {i}: radius={adaptive_r.item():.3f}, neighbors={len(neighbors)}")

        if visualize:
            plt.figure(figsize=(6, 4))
            plt.hist(adaptive_radii, bins=50, color='skyblue', edgecolor='gray')
            plt.yscale('log')
            plt.xlabel("Adaptive Radius (m)")
            plt.ylabel("Count")
            plt.title("Distribution of Adaptive Radii")
            plt.tight_layout()

        return neighbors_list, adaptive_radii

    def pad_neighbors(self, neighbors_list: List[Tensor], max_neighbors: int) -> Tuple[Tensor, Tensor]:
        padded_neighbors, mask = [], []
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
        neighbors_list, _ = self.adaptive_radius(base_radius=nn_radius, visualize=False, debug=False, n_debug_samples=0)
        padded_neighbors, mask = self.pad_neighbors(neighbors_list, max_neighbors)
        centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
        covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        clear_memory(neighbors_list, padded_neighbors, centroids, covariances)
        return eigenvalues, eigenvectors, (centered_neighbors, mask)

    def estimate_geometric_features_batched(self, neighbor_radius: float = 0.05, max_neighbors: int = 10) -> Dict[str, np.ndarray]:
        eigenvals, eigenvecs, (centered_neighbors, mask) = self.calculate_neighbor_eigens(neighbor_radius, max_neighbors)

        # Sort eigenvalues: lambda1 ≤ lambda2 ≤ lambda3
        lambda1, lambda2, lambda3 = eigenvals[:, 0], eigenvals[:, 1], eigenvals[:, 2]
        lambda_sum = eigenvals.sum(dim=1)

        # Feature calculations
        curvature = lambda1 / lambda_sum
        anisotropy = (lambda3 - lambda2) / lambda3.clamp(min=1e-9)
        planarity = (lambda2 - lambda1) / lambda3.clamp(min=1e-9)

        # Normal vectors (smallest eigenvector direction)
        normals = eigenvecs[:, :, 0]
        to_origin = -self.points_xyz
        flip_mask = (normals * to_origin).sum(dim=1) < 0
        normals[flip_mask] *= -1
        normals = normals.cpu().numpy()
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

        return {
            'curvature': curvature.cpu().numpy(),
            'anisotropy': anisotropy.cpu().numpy(),
            'planarity': planarity.cpu().numpy(),
            'nx': nx, 'ny': ny, 'nz': nz
        }



    def process_batches_with_buffer(self, dfs: List[pd.DataFrame], all_df: pd.DataFrame, buffer_pts: int, neighbor_radius: float, max_neighbors: int) -> pd.DataFrame:
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
        for df_batch in tqdm(dfs, desc="Processing batches", unit="batch"):
            buffered_df, core_mask = extract_buffered_batch_by_knn(df_batch, all_df, buffer_pts)
            batch_size = len(buffered_df)
            
            if batch_size > 30000:
                print(f"Warning: Large batch ({batch_size} points)")
            
            points_xyz = buffered_df[['X', 'Y', 'Z']].to_numpy()
            self.set_points_on_gpu(points_xyz)

            geom_feat_dict = self.estimate_geometric_features_batched(neighbor_radius, max_neighbors)

            core_df = buffered_df.loc[core_mask].copy()
            for feat_name, feat_values in geom_feat_dict.items():
                if feat_name in core_df.columns:
                    print(f"Warning: {feat_name} already exists. Overwriting.")
                core_df[feat_name] = feat_values[core_mask]

            result_dfs.append(core_df)

        geom_feat_names = ['curvature', 'anisotropy', 'planarity']
        return pd.concat(result_dfs), geom_feat_names



def infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch: int = 10_000) -> Tuple[int, int]:
    azimuth_range = points_df['azimuth'].max() - points_df['azimuth'].min()
    elevation_range = points_df['elevation'].max() - points_df['elevation'].min()
    total_points = points_df.shape[0]
    x2y_ratio = azimuth_range / elevation_range if elevation_range != 0 else 1.0
    total_batches = int(np.ceil(total_points / pts_num_per_batch))
    batch_num_x = int(np.ceil(np.sqrt(total_batches * x2y_ratio)))
    batch_num_y = int(np.ceil(total_batches / batch_num_x))
    return batch_num_x, batch_num_y


def create_xyz_spatial_batches(points_df, pts_num_per_batch: int = 10_000, max_batch_size: int = 15_000) -> List[pd.DataFrame]:
    total_points = points_df.shape[0]
    
    # Calculate 3D bounding box
    x_min, x_max = points_df['X'].min(), points_df['X'].max()
    y_min, y_max = points_df['Y'].min(), points_df['Y'].max()
    z_min, z_max = points_df['Z'].min(), points_df['Z'].max()
    
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    total_range = x_range + y_range + z_range
    
    if total_range == 0:
        return [points_df]
    
    target_batches = int(np.ceil(total_points / pts_num_per_batch))
    base_div = max(1, int(np.ceil(target_batches ** (1/3))))
    
    # Calculate divisions based on ranges
    x_div = max(1, int(np.ceil(base_div * (x_range / total_range) * 3)))
    y_div = max(1, int(np.ceil(base_div * (y_range / total_range) * 3)))
    z_div = max(1, int(np.ceil(base_div * (z_range / total_range) * 3)))
    
    # Scale down if too many divisions
    total_divisions = x_div * y_div * z_div
    if total_divisions > target_batches * 2:
        scale_factor = (target_batches * 2 / total_divisions) ** (1/3)
        x_div = max(1, int(x_div * scale_factor))
        y_div = max(1, int(y_div * scale_factor))
        z_div = max(1, int(z_div * scale_factor))
    
    # Create spatial bins
    x_bins = np.linspace(x_min, x_max + 1e-6, x_div + 1)
    y_bins = np.linspace(y_min, y_max + 1e-6, y_div + 1)
    z_bins = np.linspace(z_min, z_max + 1e-6, z_div + 1)
    
    # Assign points to spatial bins
    x_indices = np.digitize(points_df['X'], x_bins) - 1
    y_indices = np.digitize(points_df['Y'], y_bins) - 1
    z_indices = np.digitize(points_df['Z'], z_bins) - 1
    
    spatial_keys = list(zip(x_indices, y_indices, z_indices))
    points_df_copy = points_df.copy()
    points_df_copy['spatial_key'] = spatial_keys
    
    batches = []
    oversized_batches = []
    
    for spatial_key, group in points_df_copy.groupby('spatial_key'):
        group_clean = group.drop('spatial_key', axis=1)
        batch_size = len(group_clean)
        
        if batch_size == 0:
            continue
            
        if batch_size > max_batch_size:
            oversized_batches.append(group_clean)
        else:
            batches.append(group_clean)
    
    # Recursively subdivide oversized batches
    for oversized_batch in oversized_batches:
        if len(oversized_batch) > max_batch_size:
            sub_batches = create_xyz_spatial_batches(oversized_batch, 
                                                   pts_num_per_batch=max_batch_size//2, 
                                                   max_batch_size=max_batch_size)
            batches.extend(sub_batches)
        else:
            batches.append(oversized_batch)
    
    return batches

def generate_geom_feat_from_pcd(params: Dict[str, Any], points_df, output_dir, input_file_stem) -> pd.DataFrame:
    neighbor_radius = params["base_radius"]
    out_signature_str = params.get("out_signature_str", "geom_feat")
    buffer_pts = params.get("buffer_pts", 30)
    max_neighbors = params["max_neighbors"]
    pts_num_per_batch = params.get("pts_num_per_batch", 10_000)
    histogram_saveflag = params.get("histogram_saveflag", True)
    visualize = params.get("interactive_visualize", True)
    export = params.get("export", True)
    batching_method = params.get("batching_method", "angular")

    num_points = points_df.shape[0]
    geom_feature_calculator = GeometricFeatureCalculator(device='cuda')

    if num_points > 30_000:
        if batching_method == "xyz":
            dfs = create_xyz_spatial_batches(points_df, pts_num_per_batch, max_batch_size=pts_num_per_batch)
        else:
            batch_num_x, batch_num_y = infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch)
            df_grouped = points_df.groupby([
                pd.cut(points_df['azimuth'], batch_num_x), 
                pd.cut(points_df['elevation'], batch_num_y)
            ], observed=False, sort=False)
            dfs = [group for _, group in df_grouped]
        
        result_df, geom_feat_names = geom_feature_calculator.process_batches_with_buffer(
            dfs=dfs, all_df=points_df, buffer_pts=buffer_pts,
            neighbor_radius=neighbor_radius, max_neighbors=max_neighbors
        )
    else:
        result_df = points_df.copy()
        all_xyz = result_df[['X', 'Y', 'Z']].to_numpy()
        geom_feature_calculator.set_points_on_gpu(all_xyz)
        geom_feat_dict = geom_feature_calculator.estimate_geometric_features_batched(
            neighbor_radius=neighbor_radius, max_neighbors=max_neighbors
        )
        geom_feat_names = list(geom_feat_dict.keys())
        for feat_name, feat_values in geom_feat_dict.items():
            result_df[feat_name] = feat_values

    # Clean-up NaNs
    for feat_name in geom_feat_names:
        valid = result_df[feat_name].notna()
        min_val = result_df.loc[valid, feat_name].min() + (1e-4 if feat_name == 'curvature' else 0)
        result_df.loc[~valid, feat_name] = max(min_val, 0)
        if feat_name == 'curvature':
            result_df.loc[result_df[feat_name] < 0, feat_name] = min_val

    if histogram_saveflag:
        for feat_name in geom_feat_names:
            if feat_name in result_df.columns:
                get_vector_histogram(result_df[feat_name].values, output_dir, title=feat_name)

    if visualize:
        all_geom_features = []
        for feat_name in geom_feat_names:
            normalized_feat = (result_df[feat_name] - result_df[feat_name].min()) / (result_df[feat_name].max() - result_df[feat_name].min())
            all_geom_features.append(normalized_feat.to_numpy())

        all_points_xyz = result_df[['X', 'Y', 'Z']].to_numpy()
        interactive_visualize_pcd(all_points_xyz, 
                                  all_geom_features=all_geom_features, 
                                  geom_feature_names=geom_feat_names)

    if export:
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_path = save_dir / f"{input_file_stem}_{out_signature_str}.txt"
        export_results(result_df, export_path)

    return result_df



def main() -> None:
    start_time = time.time()
    profiler = MemoryProfiler()
    
    with profiler.cpu_memory_monitoring() as peak_memory:
        with profiler.gpu_memory_monitoring():
            current_config = get_config()
            
            if current_config is None:
                print("❌ Error: Configuration not loaded. Please run this script through run_3d_to_2d_pipeline.py")
                return
            
            global_params = current_config["global"]
            current_file_params = current_config["calc_geom_feature"]
            output_dir_ls = global_params["output_dir_ls"]
            input_path_ls = global_params["input_path_ls"]
            cut_percent = current_file_params["cut_percent"]
            clean_pc = current_file_params["clean_pc"]
            flip_mangrove = current_file_params["flip_mangrove"]
            dataset_name = global_params["dataset"]
            print(f"input_path_ls: {input_path_ls}")
            for input_path, output_dir in zip(input_path_ls, output_dir_ls):
                print(f'Processing {input_path.stem}...')
                points_df = read_and_clean_pcd(input_path, 
                                                cut_percent=cut_percent,
                                                clean_pc=clean_pc,
                                                dataset_name=dataset_name,
                                                flip_mangrove=flip_mangrove)
                generate_geom_feat_from_pcd(current_file_params, points_df, output_dir, input_path.stem)

    elapsed_time = time.time() - start_time
    peak_cpu_memory_mb = peak_memory[0] / (1024 * 1024)
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    print(f"Elapsed Time: {elapsed_time:.2f}s")
    print(f"Peak CPU Memory: {peak_cpu_memory_mb:.2f}MB")
    print(f"Peak GPU Memory: {peak_gpu_memory_mb:.2f}MB")

if __name__ == "__main__":
    main()