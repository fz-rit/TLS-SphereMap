"""Geometric feature calculation for point clouds using PyTorch GPU acceleration."""
import numpy as np
import pandas as pd
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from tools.preprocess_point_cloud import read_and_clean_pcd
from tools.pcd_utils import export_results, create_dir_if_not_exists
from tools.config_loader import get_config
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.points_xyz = torch.tensor(points_xyz, dtype=torch.float32, device=self.device)

    def calculate_neighbor_eigens(self, nn_radius: float, max_neighbors: int):
        num_points = self.points_xyz.shape[0]

        # Memory safety check - limit to 15K points for RTX A2000
        if num_points > 15000:
            raise RuntimeError(f"❌ Batch too large: {num_points:,} points (max 15K for GPU memory)")

        # Compute distance matrix and adaptive radii
        dist_matrix = torch.cdist(self.points_xyz, self.points_xyz)
        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        raw_adaptive_r = sorted_dists[:, 20] * 1.5  # min_pts=20, scale=1.5
        adaptive_radii = torch.clamp(raw_adaptive_r, min=nn_radius, max=0.2)
        neighbor_mask = dist_matrix <= adaptive_radii.unsqueeze(1)
        
        # Clear distance matrix immediately to free memory
        del dist_matrix, sorted_dists
        torch.cuda.empty_cache()
        
        # Create padded neighbor tensor
        padded_neighbors = torch.zeros((num_points, max_neighbors, 3), device=self.device)
        mask = torch.zeros((num_points, max_neighbors), device=self.device)
        
        point_indices, neighbor_indices = neighbor_mask.nonzero(as_tuple=True)
        sort_idx = torch.argsort(point_indices)
        sorted_point_indices = point_indices[sort_idx]
        sorted_neighbor_indices = neighbor_indices[sort_idx]
        
        unique_points, counts = torch.unique_consecutive(sorted_point_indices, return_counts=True)
        cumsum_counts = torch.cumsum(counts, dim=0)
        start_indices = torch.cat([torch.tensor([0], device=self.device), cumsum_counts[:-1]])
        
        for i, point_idx in enumerate(unique_points):
            start_idx = start_indices[i]
            end_idx = cumsum_counts[i]
            point_neighbors = sorted_neighbor_indices[start_idx:end_idx]
            count = min(len(point_neighbors), max_neighbors)
            if count > 0:
                selected_neighbors = point_neighbors[:count]
                padded_neighbors[point_idx, :count] = self.points_xyz[selected_neighbors]
                mask[point_idx, :count] = 1.0
        
        # Clear neighbor finding arrays
        del neighbor_mask, point_indices, neighbor_indices, sort_idx
        torch.cuda.empty_cache()
        
        # Compute covariance and eigenvalues
        centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        centered_neighbors = padded_neighbors - centroids.unsqueeze(1)
        covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        clear_memory(padded_neighbors, centroids, covariances)
        return eigenvalues, eigenvectors, (centered_neighbors, mask)

    def estimate_geometric_features_batched(self, neighbor_radius: float = 0.05, max_neighbors: int = 10) -> Dict[str, np.ndarray]:
        eigenvals, eigenvecs, (centered_neighbors, mask) = self.calculate_neighbor_eigens(neighbor_radius, max_neighbors)

        lambda1, lambda2, lambda3 = eigenvals[:, 0], eigenvals[:, 1], eigenvals[:, 2]
        lambda_sum = eigenvals.sum(dim=1)

        curvature = lambda1 / lambda_sum
        anisotropy = (lambda3 - lambda2) / lambda3.clamp(min=1e-9)
        planarity = (lambda2 - lambda1) / lambda3.clamp(min=1e-9)

        normals = eigenvecs[:, :, 0]
        to_origin = -self.points_xyz
        flip_mask = (normals * to_origin).sum(dim=1) < 0
        normals[flip_mask] *= -1
        normals = normals.cpu().numpy()

        return {
            'curvature': curvature.cpu().numpy(),
            'anisotropy': anisotropy.cpu().numpy(),
            'planarity': planarity.cpu().numpy(),
            'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2]
        }

    def process_batches_with_buffer(self, dfs: List[pd.DataFrame], all_df: pd.DataFrame, 
                                  buffer_pts: int, neighbor_radius: float, max_neighbors: int,
                                  max_workers: int = 4) -> Tuple[pd.DataFrame, List[str]]:
        # Process sequentially to avoid GPU conflicts
        result_dfs = []
        total_start = time.time()
        
        print(f"🚀 Processing {len(dfs)} batches sequentially (GPU-safe)...")
        
        all_points = all_df[['X', 'Y', 'Z']].to_numpy()
        kdtree = KDTree(all_points)
        
        for batch_idx, df_batch in enumerate(tqdm(dfs, desc="GPU batches", unit="batch")):
            batch_start = time.time()
            
            # Extract buffered batch
            core_points = df_batch[['X', 'Y', 'Z']].to_numpy()
            indices = kdtree.query(core_points, k=buffer_pts, return_distance=False)
            neighbor_indices = set(indices.flatten())
            buffered_df = all_df.iloc[list(neighbor_indices)].copy()
            core_mask = buffered_df.index.isin(df_batch.index)
            
            # Process on GPU
            points_xyz = buffered_df[['X', 'Y', 'Z']].to_numpy()
            self.set_points_on_gpu(points_xyz)
            geom_feat_dict = self.estimate_geometric_features_batched(neighbor_radius, max_neighbors)

            # Extract core results
            core_df = buffered_df.loc[core_mask].copy()
            for feat_name, feat_values in geom_feat_dict.items():
                core_df[feat_name] = feat_values[core_mask]

            result_dfs.append(core_df)
            batch_time = time.time() - batch_start
            
            if batch_idx % 100 == 0:
                print(f"✓ Batch {batch_idx}: {len(core_df):,} points in {batch_time:.1f}s")
            
            clear_memory()

        total_time = time.time() - total_start
        total_points = sum(len(df) for df in result_dfs)
        
        print(f"🎉 Sequential processing completed: {total_points:,} points in {total_time:.1f}s")
        print(f"⚡ Average: {total_points/total_time:.0f} points/sec")
        
        geom_feat_names = ['curvature', 'anisotropy', 'planarity', 'nx', 'ny', 'nz']
        return pd.concat(result_dfs, ignore_index=True), geom_feat_names



def infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch: int = 10_000) -> Tuple[int, int]:
    azimuth_range = points_df['azimuth'].max() - points_df['azimuth'].min()
    elevation_range = points_df['elevation'].max() - points_df['elevation'].min()
    total_points = points_df.shape[0]
    x2y_ratio = azimuth_range / elevation_range if elevation_range != 0 else 1.0
    total_batches = int(np.ceil(total_points / pts_num_per_batch))
    batch_num_x = int(np.ceil(np.sqrt(total_batches * x2y_ratio)))
    batch_num_y = int(np.ceil(total_batches / batch_num_x))
    return batch_num_x, batch_num_y


def create_xyz_spatial_batches(points_df, pts_num_per_batch: int = 10_000, max_batch_size: int = 12_000) -> List[pd.DataFrame]:
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
    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU required but not available")
    
    neighbor_radius = params["base_radius"]
    max_neighbors = params["max_neighbors"]
    pts_num_per_batch = params.get("pts_num_per_batch", 5000)  # Smaller batches for parallel
    max_workers = params.get("max_workers", 4)
    buffer_pts = params.get("buffer_pts", 30)
    batching_method = params.get("batching_method", "xyz")
    export = params.get("export", True)
    out_signature_str = params.get("out_signature_str", "geom_feat")

    num_points = points_df.shape[0]
    print(f"📊 Processing {num_points:,} points (GPU: {torch.cuda.get_device_name()})")
    
    geom_feature_calculator = GeometricFeatureCalculator(device='cuda')
    start_time = time.time()

    if num_points > 20_000:  # Lower threshold for parallel processing
        print(f"🔄 Creating batches (target: {pts_num_per_batch:,} points/batch)")
        
        if batching_method == "xyz":
            dfs = create_xyz_spatial_batches(points_df, pts_num_per_batch, max_batch_size=pts_num_per_batch)
        else:
            batch_num_x, batch_num_y = infer_batch_num_azimuth_elevation(points_df, pts_num_per_batch)
            df_grouped = points_df.groupby([
                pd.cut(points_df['azimuth'], batch_num_x), 
                pd.cut(points_df['elevation'], batch_num_y)
            ], observed=False, sort=False)
            dfs = [group for _, group in df_grouped]
        
        print(f"📦 Created {len(dfs)} batches (avg: {num_points//len(dfs):,} points/batch)")
        
        # Parallel batch processing
        result_df, geom_feat_names = geom_feature_calculator.process_batches_with_buffer(
            dfs=dfs, all_df=points_df, buffer_pts=buffer_pts,
            neighbor_radius=neighbor_radius, max_neighbors=max_neighbors,
            max_workers=max_workers
        )
    else:
        print("💨 Direct processing (small dataset)")
        result_df = points_df.copy()
        all_xyz = result_df[['X', 'Y', 'Z']].to_numpy()
        geom_feature_calculator.set_points_on_gpu(all_xyz)
        geom_feat_dict = geom_feature_calculator.estimate_geometric_features_batched(
            neighbor_radius=neighbor_radius, max_neighbors=max_neighbors
        )
        geom_feat_names = list(geom_feat_dict.keys())
        for feat_name, feat_values in geom_feat_dict.items():
            result_df[feat_name] = feat_values

    # Clean NaNs efficiently
    for feat_name in geom_feat_names[:3]:  # Only geometric features
        mask = result_df[feat_name].isna()
        if mask.any():
            result_df.loc[mask, feat_name] = 0.0

    processing_time = time.time() - start_time
    points_per_sec = num_points / processing_time
    print(f"✅ Completed: {processing_time:.1f}s ({points_per_sec:.0f} points/sec)")

    if export:
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_path = save_dir / f"{input_file_stem}_{out_signature_str}.txt"
        export_results(result_df, export_path)
        print(f"💾 Saved: {export_path}")

    return result_df



def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU required but not available")
    
    start_time = time.time()
    current_config = get_config()
    
    if current_config is None:
        raise RuntimeError("❌ Configuration not loaded. Run through run_3d_to_2d_pipeline.py")
    
    global_params = current_config["global"]
    current_file_params = current_config["calc_geom_feature"]
    output_dir_ls = global_params["output_dir_ls"]
    input_path_ls = global_params["input_path_ls"]
    dataset_name = global_params["dataset"]
    
    print(f"🚀 GPU Geometric Feature Calculator")
    print(f"📂 Processing {len(input_path_ls)} files from {dataset_name}")
    
    total_points = 0
    for input_path, output_dir in zip(input_path_ls, output_dir_ls):
        file_start = time.time()
        print(f"\n🔹 Processing: {input_path.stem}")
        
        points_df = read_and_clean_pcd(
            input_path, 
            cut_percent=current_file_params["cut_percent"],
            clean_pc=current_file_params["clean_pc"],
            dataset_name=dataset_name,
            flip_mangrove=current_file_params["flip_mangrove"]
        )
        
        result_df = generate_geom_feat_from_pcd(current_file_params, points_df, output_dir, input_path.stem)
        
        file_time = time.time() - file_start
        file_points = len(result_df)
        total_points += file_points
        print(f"🎯 File completed: {file_points:,} points in {file_time:.1f}s")

    total_time = time.time() - start_time
    avg_speed = total_points / total_time
    
    print(f"\n🎉 All files completed!")
    print(f"📊 Total: {total_points:,} points in {total_time:.1f}s")
    print(f"⚡ Average speed: {avg_speed:.0f} points/sec")
    print(f"🔋 Peak GPU Memory: {torch.cuda.max_memory_allocated() / (1024**3):.1f}GB")

if __name__ == "__main__":
    main()