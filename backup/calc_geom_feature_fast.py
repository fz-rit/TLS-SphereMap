"""
Fast geometric feature calculation using Open3D and optimized algorithms.
This is a complete rewrite focusing on performance for large point clouds.
"""
import numpy as np
import pandas as pd
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import gc

# Use Open3D for efficient geometric computations
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Install with: pip install open3d")

from tools.preprocess_point_cloud import read_and_clean_pcd
from tools.pcd_utils import export_results, MemoryProfiler, create_dir_if_not_exists
from tools.config_loader import get_config


class FastGeometricFeatureCalculator:
    """
    High-performance geometric feature calculator using Open3D's optimized algorithms.
    Designed for large point clouds with millions of points.
    """
    
    def __init__(self):
        self.pcd = None
        self.points = None
        self.tree = None
        
    def set_points(self, points_xyz: np.ndarray):
        """Set points and build spatial index for fast queries."""
        self.points = points_xyz.astype(np.float64)  # Open3D prefers float64
        
        # Create Open3D point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Build KDTree for fast neighbor search
        self.tree = o3d.geometry.KDTreeFlann(self.pcd)
        
    def estimate_normals_fast(self, radius: float = 0.03, max_nn: int = 30):
        """
        Fast normal estimation using Open3D's optimized implementation.
        Uses hybrid search with both radius and max neighbors.
        """
        # Open3D's highly optimized normal estimation
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        
        # Orient normals consistently (toward origin)
        self.pcd.orient_normals_to_align_with_direction(
            orientation_reference=np.array([0., 0., 0.])
        )
        
        normals = np.asarray(self.pcd.normals)
        return normals
    
    def calculate_local_features_vectorized(self, radius: float = 0.03, max_nn: int = 30):
        """
        Vectorized calculation of geometric features using efficient neighbor search.
        Much faster than per-point eigenvalue decomposition.
        """
        n_points = len(self.points)
        
        # Pre-allocate arrays
        curvatures = np.zeros(n_points)
        anisotropies = np.zeros(n_points)
        planarities = np.zeros(n_points)
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        
        print(f"Processing {n_points} points in chunks of {chunk_size}...")
        
        for start_idx in tqdm(range(0, n_points, chunk_size), desc="Computing features"):
            end_idx = min(start_idx + chunk_size, n_points)
            chunk_size_actual = end_idx - start_idx
            
            # Batch neighbor search for chunk
            chunk_neighbors = []
            chunk_neighbor_counts = []
            
            for i in range(start_idx, end_idx):
                # Hybrid search: radius with max neighbors limit
                [k, idx, dist] = self.tree.search_hybrid_vector_3d(self.points[i], radius, max_nn)
                
                if k > 3:  # Need at least 3 neighbors for covariance
                    neighbor_points = self.points[idx[1:k]]  # Exclude self
                    chunk_neighbors.append(neighbor_points)
                    chunk_neighbor_counts.append(k-1)
                else:
                    # Fallback: use k-nearest neighbors if too few in radius
                    [k, idx, _] = self.tree.search_knn_vector_3d(self.points[i], 10)
                    neighbor_points = self.points[idx[1:]]  # Exclude self
                    chunk_neighbors.append(neighbor_points)
                    chunk_neighbor_counts.append(k-1)
            
            # Vectorized covariance computation for chunk
            for local_idx, neighbors in enumerate(chunk_neighbors):
                global_idx = start_idx + local_idx
                
                if len(neighbors) < 3:
                    continue
                    
                # Center the neighbors
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                
                # Covariance matrix
                cov = np.cov(centered.T)
                
                # Eigenvalues (sorted ascending)
                eigenvals = np.linalg.eigvalsh(cov)
                eigenvals = np.sort(eigenvals)
                
                # Geometric features
                if eigenvals[2] > 1e-10:  # Avoid division by zero
                    curvatures[global_idx] = eigenvals[0] / eigenvals.sum()
                    anisotropies[global_idx] = (eigenvals[2] - eigenvals[1]) / eigenvals[2]
                    planarities[global_idx] = (eigenvals[1] - eigenvals[0]) / eigenvals[2]
        
        return curvatures, anisotropies, planarities
    
    def estimate_all_features(self, radius: float = 0.03, max_nn: int = 30):
        """
        Complete geometric feature estimation pipeline.
        Returns all features in one efficient pass.
        """
        print(f"Estimating geometric features with radius={radius:.3f}, max_nn={max_nn}")
        
        # Fast normal estimation
        normals = self.estimate_normals_fast(radius, max_nn)
        
        # Fast local feature calculation
        curvatures, anisotropies, planarities = self.calculate_local_features_vectorized(radius, max_nn)
        
        return {
            'curvature': curvatures,
            'anisotropy': anisotropies, 
            'planarity': planarities,
            'nx': normals[:, 0],
            'ny': normals[:, 1],
            'nz': normals[:, 2]
        }


def adaptive_radius_simple(points: np.ndarray, base_radius: float = 0.01) -> float:
    """
    Simple density-based radius calculation without expensive operations.
    """
    n_points = len(points)
    
    # Quick bounding box volume
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    volume = np.prod(maxs - mins)
    
    if volume <= 0:
        return base_radius
    
    # Point density
    density = n_points / volume
    density_threshold = 1.0 / (0.02 ** 3)  # 125,000 pts/m³
    
    if density >= density_threshold:
        return 0.03  # High density
    else:
        # Scale radius for low density
        scale_factor = np.sqrt(density_threshold / density)
        return min(base_radius * scale_factor, 0.2)


def process_large_pointcloud_chunked(points_df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Process very large point clouds using spatial chunking and parallel processing.
    """
    base_radius = params["base_radius"]
    max_neighbors = params.get("max_neighbors", 30)
    chunk_overlap = params.get("buffer_pts", 30) * 0.001  # Convert to meters
    
    points_xyz = points_df[['X', 'Y', 'Z']].to_numpy()
    n_points = len(points_xyz)
    
    print(f"Processing large point cloud with {n_points:,} points using chunked approach")
    
    # Determine chunk size based on available memory
    max_chunk_size = 100_000  # Adjust based on your RAM
    
    # Calculate spatial grid for chunking
    mins = np.min(points_xyz, axis=0)
    maxs = np.max(points_xyz, axis=0)
    ranges = maxs - mins
    
    # Estimate grid divisions
    target_chunks = max(1, n_points // max_chunk_size)
    grid_size = int(np.ceil(target_chunks ** (1/3)))
    
    # Create spatial grid
    x_edges = np.linspace(mins[0] - chunk_overlap, maxs[0] + chunk_overlap, grid_size + 1)
    y_edges = np.linspace(mins[1] - chunk_overlap, maxs[1] + chunk_overlap, grid_size + 1)
    z_edges = np.linspace(mins[2] - chunk_overlap, maxs[2] + chunk_overlap, grid_size + 1)
    
    result_features = []
    processed_indices = set()
    
    calculator = FastGeometricFeatureCalculator()
    radius = adaptive_radius_simple(points_xyz, base_radius)
    
    print(f"Using adaptive radius: {radius:.4f}m")
    print(f"Processing in {grid_size}³ = {grid_size**3} spatial chunks")
    
    for i in tqdm(range(grid_size), desc="Grid X"):
        for j in range(grid_size):
            for k in range(grid_size):
                # Define chunk boundaries
                x_min, x_max = x_edges[i], x_edges[i+1]
                y_min, y_max = y_edges[j], y_edges[j+1]
                z_min, z_max = z_edges[k], z_edges[k+1]
                
                # Find points in chunk
                mask = (
                    (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] < x_max) &
                    (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] < y_max) &
                    (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] < z_max)
                )
                
                chunk_indices = np.where(mask)[0]
                
                if len(chunk_indices) < 10:  # Skip tiny chunks
                    continue
                
                # Core points (without overlap buffer)
                core_mask = (
                    (points_xyz[:, 0] >= x_min + chunk_overlap) & (points_xyz[:, 0] < x_max - chunk_overlap) &
                    (points_xyz[:, 1] >= y_min + chunk_overlap) & (points_xyz[:, 1] < y_max - chunk_overlap) &
                    (points_xyz[:, 2] >= z_min + chunk_overlap) & (points_xyz[:, 2] < z_max - chunk_overlap)
                )
                core_indices = np.where(core_mask)[0]
                core_indices = [idx for idx in core_indices if idx not in processed_indices]
                
                if len(core_indices) < 5:
                    continue
                
                # Process chunk
                chunk_points = points_xyz[chunk_indices]
                calculator.set_points(chunk_points)
                
                features = calculator.estimate_all_features(radius, max_neighbors)
                
                # Map back to core points
                core_local_indices = []
                for core_idx in core_indices:
                    local_idx = np.where(chunk_indices == core_idx)[0]
                    if len(local_idx) > 0:
                        core_local_indices.append(local_idx[0])
                
                if core_local_indices:
                    core_features = {
                        'index': core_indices,
                        'curvature': features['curvature'][core_local_indices],
                        'anisotropy': features['anisotropy'][core_local_indices],
                        'planarity': features['planarity'][core_local_indices],
                        'nx': features['nx'][core_local_indices],
                        'ny': features['ny'][core_local_indices],
                        'nz': features['nz'][core_local_indices]
                    }
                    result_features.append(core_features)
                    processed_indices.update(core_indices)
    
    # Combine results
    if result_features:
        all_indices = np.concatenate([f['index'] for f in result_features])
        all_features = {
            'curvature': np.concatenate([f['curvature'] for f in result_features]),
            'anisotropy': np.concatenate([f['anisotropy'] for f in result_features]),
            'planarity': np.concatenate([f['planarity'] for f in result_features]),
            'nx': np.concatenate([f['nx'] for f in result_features]),
            'ny': np.concatenate([f['ny'] for f in result_features]),
            'nz': np.concatenate([f['nz'] for f in result_features])
        }
        
        # Create result DataFrame
        result_df = points_df.iloc[all_indices].copy()
        for feat_name, feat_values in all_features.items():
            result_df[feat_name] = feat_values
            
        return result_df
    else:
        return points_df.copy()


def generate_geom_feat_from_pcd_fast(params: Dict[str, Any], points_df: pd.DataFrame, 
                                   output_dir: Path, input_file_stem: str) -> pd.DataFrame:
    """
    Fast geometric feature calculation for large point clouds.
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for fast processing. Install with: pip install open3d")
    
    # Parameters
    base_radius = params["base_radius"]
    max_neighbors = params.get("max_neighbors", 30)
    export = params.get("export", True)
    out_signature_str = params.get("out_signature_str", "geom_feat")
    
    n_points = len(points_df)
    print(f"Processing {n_points:,} points with fast algorithm")
    
    # Choose processing strategy based on size
    if n_points > 500_000:  # Very large
        print("Using chunked processing for very large point cloud")
        result_df = process_large_pointcloud_chunked(points_df, params)
    else:  # Medium to large
        print("Using optimized single-pass processing")
        points_xyz = points_df[['X', 'Y', 'Z']].to_numpy()
        
        calculator = FastGeometricFeatureCalculator()
        calculator.set_points(points_xyz)
        
        radius = adaptive_radius_simple(points_xyz, base_radius)
        features = calculator.estimate_all_features(radius, max_neighbors)
        
        result_df = points_df.copy()
        for feat_name, feat_values in features.items():
            result_df[feat_name] = feat_values
    
    # Clean up NaN values
    feature_names = ['curvature', 'anisotropy', 'planarity', 'nx', 'ny', 'nz']
    for feat_name in feature_names:
        if feat_name in result_df.columns:
            valid_mask = np.isfinite(result_df[feat_name])
            if not valid_mask.all():
                print(f"Cleaning {(~valid_mask).sum()} invalid values in {feat_name}")
                if feat_name in ['curvature', 'anisotropy', 'planarity']:
                    result_df.loc[~valid_mask, feat_name] = 0.0
                else:  # normals
                    result_df.loc[~valid_mask, feat_name] = 0.0
    
    # Export results
    if export:
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_path = save_dir / f"{input_file_stem}_{out_signature_str}.txt"
        export_results(result_df, export_path)
        print(f"Results saved to {export_path}")
    
    return result_df


def main() -> None:
    """Main function with performance monitoring."""
    start_time = time.time()
    
    if not HAS_OPEN3D:
        print("❌ Open3D not available. Please install: pip install open3d")
        return
    
    current_config = get_config()
    if current_config is None:
        print("❌ Error: Configuration not loaded. Please run through run_3d_to_2d_pipeline.py")
        return
    
    global_params = current_config["global"]
    current_file_params = current_config["calc_geom_feature"]
    output_dir_ls = global_params["output_dir_ls"]
    input_path_ls = global_params["input_path_ls"]
    
    # Processing parameters
    cut_percent = current_file_params["cut_percent"]
    clean_pc = current_file_params["clean_pc"]
    flip_mangrove = current_file_params["flip_mangrove"]
    dataset_name = global_params["dataset"]
    
    print(f"Fast geometric feature calculation starting...")
    print(f"Processing {len(input_path_ls)} files")
    
    for input_path, output_dir in zip(input_path_ls, output_dir_ls):
        print(f"\n🚀 Processing {input_path.stem}...")
        
        # Load and preprocess
        points_df = read_and_clean_pcd(
            input_path,
            cut_percent=cut_percent,
            clean_pc=clean_pc,
            dataset_name=dataset_name,
            flip_mangrove=flip_mangrove
        )
        
        # Process with fast algorithm
        file_start = time.time()
        result_df = generate_geom_feat_from_pcd_fast(
            current_file_params, points_df, output_dir, input_path.stem
        )
        file_time = time.time() - file_start
        
        print(f"✅ Completed {input_path.stem} in {file_time:.1f}s")
        print(f"   Processed {len(result_df):,} points")
        
        # Memory cleanup
        del points_df, result_df
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"\n🎉 All files completed in {total_time:.1f}s")


if __name__ == "__main__":
    main()
