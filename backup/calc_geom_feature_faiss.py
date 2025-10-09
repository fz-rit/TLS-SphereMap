"""
Ultra-fast geometric feature calculation using FAISS for neighbor search.
Alternative implementation that doesn't require Open3D.
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import gc

# Try to import FAISS for ultra-fast neighbor search
try:
    import faiss
    HAS_FAISS = True
    
    # Check for GPU support
    try:
        faiss.StandardGpuResources()
        HAS_FAISS_GPU = True
        print("FAISS GPU support available")
    except:
        HAS_FAISS_GPU = False
        print("FAISS CPU-only mode")
        
except ImportError:
    HAS_FAISS = False
    HAS_FAISS_GPU = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from tools.preprocess_point_cloud import read_and_clean_pcd
from tools.pcd_utils import export_results, create_dir_if_not_exists
from tools.config_loader import get_config


def check_gpu_support() -> bool:
    """Check if FAISS GPU support is available."""
    return HAS_FAISS_GPU


def build_faiss_index(points: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """Build FAISS index for ultra-fast neighbor search with optional GPU acceleration."""
    # Ensure C-contiguous float32 array (FAISS requirement)
    points_f32 = np.ascontiguousarray(points.astype(np.float32))
    n_points, dim = points_f32.shape
    
    print(f"Building FAISS index for {n_points:,} points...")
    
    # Choose index type based on dataset size
    if n_points < 100_000:
        # Small dataset: use exact search
        index = faiss.IndexFlatL2(dim)
    else:
        # Large dataset: use approximate search for speed
        # IVF with 4*sqrt(n) clusters is a good rule of thumb
        nlist = min(4096, max(256, int(4 * np.sqrt(n_points))))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        
        # Train the index on the data
        print(f"Training IVF index with {nlist} clusters...")
        index.train(points_f32)
    
    # Move to GPU if available and requested
    if use_gpu and HAS_FAISS_GPU:
        try:
            gpu_res = faiss.StandardGpuResources()
            
            # Configure GPU memory (optional)
            gpu_res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
            
            # Move index to GPU
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)  # Use GPU 0
            print("Index moved to GPU")
        except Exception as e:
            print(f"GPU initialization failed, using CPU: {e}")
            use_gpu = False
    
    # Add points to index
    index.add(points_f32)
    
    # Set search parameters for IVF
    if hasattr(index, 'nprobe'):
        # Search more clusters for better accuracy
        index.nprobe = min(nlist // 4, 128) if n_points >= 100_000 else nlist
        print(f"IVF search will probe {index.nprobe} clusters")
    
    return index


def compute_geometric_features_batch(points: np.ndarray, neighbor_indices: np.ndarray, 
                                   valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ultra-fast vectorized computation of geometric features using optimized batch processing.
    """
    n_points = len(points)
    curvatures = np.zeros(n_points)
    anisotropies = np.zeros(n_points)
    planarities = np.zeros(n_points)
    normals = np.zeros((n_points, 3))
    
    try:
        # Use the optimized batch covariance computation
        eigenvals, eigenvecs, valid_indices = compute_batch_covariance_fast(
            points, neighbor_indices, valid_mask
        )
        
        if len(valid_indices) == 0:
            return {
                'curvature': curvatures, 'anisotropy': anisotropies, 'planarity': planarities,
                'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2]
            }
        
        # Vectorized geometric features calculation
        eigenval_sums = np.sum(eigenvals, axis=1)
        
        # Create masks for valid computations (avoid division by zero)
        valid_sums = eigenval_sums > 1e-10
        valid_lambda2 = eigenvals[:, 2] > 1e-10
        fully_valid = valid_sums & valid_lambda2
        
        # Vectorized feature computation for all valid points
        if np.any(fully_valid):
            # Curvature: λ₀ / (λ₀ + λ₁ + λ₂)
            curvatures[valid_indices[fully_valid]] = (
                eigenvals[fully_valid, 0] / eigenval_sums[fully_valid]
            )
            
            # Anisotropy: (λ₂ - λ₁) / λ₂
            anisotropies[valid_indices[fully_valid]] = (
                (eigenvals[fully_valid, 2] - eigenvals[fully_valid, 1]) / 
                eigenvals[fully_valid, 2]
            )
            
            # Planarity: (λ₁ - λ₀) / λ₂  
            planarities[valid_indices[fully_valid]] = (
                (eigenvals[fully_valid, 1] - eigenvals[fully_valid, 0]) / 
                eigenvals[fully_valid, 2]
            )
        
        # Vectorized normal vector computation and orientation
        normal_vectors = eigenvecs[:, :, 0]  # Smallest eigenvectors
        query_points = points[valid_indices]
        
        # Orient all normals toward origin simultaneously
        to_origin = -query_points
        dot_products = np.sum(normal_vectors * to_origin, axis=1)
        
        # Flip normals where needed (vectorized)
        flip_mask = dot_products < 0
        normal_vectors[flip_mask] *= -1
        
        # Assign to output array
        normals[valid_indices] = normal_vectors
        
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Fast batch computation failed ({e}), using fallback")
        # Fallback to individual processing for robustness
        valid_indices = np.where(valid_mask)[0]
        
        for i in valid_indices:
            try:
                neighbors = points[neighbor_indices[i]]
                if len(neighbors) < 4:
                    continue
                    
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                cov = np.cov(centered.T)
                eigenvals_local, eigenvecs_local = np.linalg.eigh(cov)
                
                # Sort eigenvalues
                sort_idx = np.argsort(eigenvals_local)
                eigenvals_local = eigenvals_local[sort_idx]
                eigenvecs_local = eigenvecs_local[:, sort_idx]
                
                if eigenvals_local[2] > 1e-10:
                    curvatures[i] = eigenvals_local[0] / eigenvals_local.sum()
                    anisotropies[i] = (eigenvals_local[2] - eigenvals_local[1]) / eigenvals_local[2]
                    planarities[i] = (eigenvals_local[1] - eigenvals_local[0]) / eigenvals_local[2]
                
                # Normal vector orientation
                normal = eigenvecs_local[:, 0]
                to_origin_local = -points[i]
                if np.dot(normal, to_origin_local) < 0:
                    normal = -normal
                normals[i] = normal
                
            except:
                continue
    
    return {
        'curvature': curvatures,
        'anisotropy': anisotropies,
        'planarity': planarities,
        'nx': normals[:, 0],
        'ny': normals[:, 1],
        'nz': normals[:, 2]
    }


def compute_batch_covariance_fast(points: np.ndarray, neighbor_indices: np.ndarray, 
                                  valid_mask: np.ndarray) -> tuple:
    """
    Ultra-fast batch covariance computation using advanced NumPy vectorization.
    Returns eigenvalues and eigenvectors for all valid points simultaneously.
    """
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return np.array([]), np.array([]), valid_indices
    
    # Get all neighbor coordinates at once - shape: (n_valid, max_neighbors, 3)
    valid_neighbors = neighbor_indices[valid_indices]
    neighbor_coords = points[valid_neighbors]
    
    # Compute centroids for all neighborhoods - shape: (n_valid, 3)
    centroids = np.mean(neighbor_coords, axis=1)
    
    # Center all coordinates - shape: (n_valid, max_neighbors, 3)
    centered = neighbor_coords - centroids[:, np.newaxis, :]
    
    # Batch covariance computation using einsum for maximum efficiency
    # This computes C = (1/n) * X^T @ X for all points simultaneously
    n_neighbors = neighbor_coords.shape[1]
    batch_cov = np.einsum('bij,bik->bjk', centered, centered) / n_neighbors
    
    # Batch eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(batch_cov)
    
    # Sort eigenvalues in ascending order for all points
    sort_indices = np.argsort(eigenvals, axis=1)
    eigenvals = np.take_along_axis(eigenvals, sort_indices, axis=1)
    eigenvecs = np.take_along_axis(eigenvecs, sort_indices[:, np.newaxis, :], axis=2)
    
    return eigenvals, eigenvecs, valid_indices


def process_with_faiss(points: np.ndarray, radius: float, max_neighbors: int = 50, 
                      use_gpu: bool = True) -> Dict[str, np.ndarray]:
    """
    Process point cloud using FAISS for ultra-fast neighbor search with GPU acceleration.
    """
    n_points = len(points)
    print(f"Processing {n_points:,} points with FAISS...")
    
    # Build index with GPU support
    index = build_faiss_index(points, use_gpu=use_gpu)
    
    # Search for neighbors within radius
    print(f"Searching neighbors within radius {radius:.4f}...")
    
    # FAISS range search (all points at once) - ensure C-contiguous
    points_f32 = np.ascontiguousarray(points.astype(np.float32))
    
    try:
        lims, D, I = index.range_search(points_f32, radius ** 2)  # squared distance
        print("Using range search results")
    except Exception as e:
        print(f"Range search failed: {e}, falling back to k-NN")
        # Vectorized fallback to k-nearest neighbor search
        k = min(max_neighbors + 1, n_points)
        D, I = index.search(points_f32, k)
        
        # Vectorized distance filtering
        radius_sq = radius ** 2
        valid_distances = D <= radius_sq
        
        # Convert to range search format efficiently
        lims = np.zeros(n_points + 1, dtype=np.int32)
        valid_counts = np.sum(valid_distances, axis=1)
        lims[1:] = np.cumsum(valid_counts)
        
        # Flatten valid neighbors
        I_filtered = []
        for i in range(n_points):
            valid_mask = valid_distances[i]
            I_filtered.extend(I[i][valid_mask])
        I = np.array(I_filtered, dtype=np.int32)
    
    # Vectorized neighbor list processing
    neighbor_lists = []
    valid_mask = np.zeros(n_points, dtype=bool)
    
    # Process all points at once
    neighbor_counts = lims[1:] - lims[:-1]
    has_neighbors = neighbor_counts > 1  # More than just self
    
    for i in range(n_points):
        start_idx = lims[i]
        end_idx = lims[i + 1]
        
        if has_neighbors[i]:
            neighbors = I[start_idx:end_idx]
            
            # Vectorized self-removal and limiting
            neighbors = neighbors[neighbors != i]
            if len(neighbors) > max_neighbors:
                # Use faster random choice without replacement
                indices = np.random.choice(len(neighbors), max_neighbors, replace=False)
                neighbors = neighbors[indices]
            
            # Include self at the beginning
            neighbors = np.concatenate([[i], neighbors])
            neighbor_lists.append(neighbors)
            valid_mask[i] = True
        else:
            # Vectorized fallback: k-nearest neighbors
            k = min(max_neighbors + 1, n_points)
            single_point = points_f32[i:i+1]
            _, knn_indices = index.search(single_point, k)
            neighbors = knn_indices[0]
            neighbor_lists.append(neighbors)
            valid_mask[i] = True
    
    # Vectorized padding of neighbor lists for efficient batch processing
    if neighbor_lists:
        max_len = max(len(nl) for nl in neighbor_lists)
        neighbor_indices = np.zeros((n_points, max_len), dtype=np.int32)
        
        # Vectorized padding - use numpy operations instead of loops
        for i, neighbors in enumerate(neighbor_lists):
            n_neighbors = len(neighbors)
            neighbor_indices[i, :n_neighbors] = neighbors
            if n_neighbors < max_len:
                # Pad with last neighbor
                neighbor_indices[i, n_neighbors:] = neighbors[-1]
    else:
        # Fallback case
        neighbor_indices = np.zeros((n_points, max_neighbors + 1), dtype=np.int32)
        for i in range(n_points):
            neighbor_indices[i] = i  # Just self-reference
    
    print("Computing geometric features...")
    features = compute_geometric_features_batch(points, neighbor_indices, valid_mask)
    
    return features


def process_chunked_faiss(points: np.ndarray, radius: float, max_neighbors: int = 50, 
                         chunk_size: int = 100000, use_gpu: bool = True) -> Dict[str, np.ndarray]:
    """
    Process very large point clouds in chunks using FAISS with GPU acceleration.
    """
    n_points = len(points)
    print(f"Processing {n_points:,} points in chunks of {chunk_size:,} with GPU={use_gpu}")
    
    # Initialize result arrays
    all_features = {
        'curvature': np.zeros(n_points),
        'anisotropy': np.zeros(n_points),
        'planarity': np.zeros(n_points),
        'nx': np.zeros(n_points),
        'ny': np.zeros(n_points),
        'nz': np.zeros(n_points)
    }
    
    # Build global index once with GPU support
    index = build_faiss_index(points, use_gpu=use_gpu)
    
    # Process in chunks
    for start_idx in tqdm(range(0, n_points, chunk_size), desc="Processing chunks"):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points[start_idx:end_idx]
        
        # Search neighbors for chunk - ensure C-contiguous
        points_f32 = np.ascontiguousarray(chunk_points.astype(np.float32))
        
        # Vectorized chunk processing
        chunk_size_actual = end_idx - start_idx
        
        try:
            lims, D, I = index.range_search(points_f32, radius ** 2)
            
            # Vectorized neighbor extraction and processing
            neighbor_lists = []
            valid_mask = np.zeros(chunk_size_actual, dtype=bool)
            
            # Process all chunk points at once
            neighbor_counts = lims[1:] - lims[:-1]
            has_enough_neighbors = neighbor_counts > 1
            
            for i in range(chunk_size_actual):
                start_search = lims[i]
                end_search = lims[i + 1]
                
                if has_enough_neighbors[i]:
                    neighbors = I[start_search:end_search]
                    # Efficient unique and limiting
                    neighbors = np.unique(neighbors)
                    if len(neighbors) > max_neighbors:
                        # Use numpy's random choice for efficiency
                        indices = np.random.choice(len(neighbors), max_neighbors, replace=False)
                        neighbors = neighbors[indices]
                    neighbor_lists.append(neighbors)
                    valid_mask[i] = True
                else:
                    # Vectorized fallback
                    global_idx = start_idx + i
                    k = min(max_neighbors, n_points)
                    single_point = np.ascontiguousarray(points[global_idx:global_idx+1].astype(np.float32))
                    _, knn_indices = index.search(single_point, k)
                    neighbor_lists.append(knn_indices[0])
                    valid_mask[i] = True
                    
        except:
            # Fallback to k-nearest neighbor with vectorized processing
            k = min(max_neighbors + 1, n_points)
            D, I = index.search(points_f32, k)
            
            # Create simplified neighbor structure
            neighbor_lists = []
            valid_mask = np.ones(chunk_size_actual, dtype=bool)  # All valid in k-NN
            
            # Vectorized processing of k-NN results
            chunk_indices = np.arange(start_idx, end_idx)[:, np.newaxis]  # Shape: (chunk_size, 1)
            I_flat = I.flatten()
            lims = np.arange(0, (chunk_size_actual + 1) * k, k)
            
            for i in range(chunk_size_actual):
                start_search = lims[i]
                end_search = lims[i + 1]
                neighbors = I_flat[start_search:end_search]
                neighbor_lists.append(neighbors)
        
        # Vectorized feature computation preparation
        if neighbor_lists:
            max_len = max(len(nl) for nl in neighbor_lists)
            neighbor_indices = np.zeros((chunk_size_actual, max_len), dtype=np.int32)
            
            # Efficient padding
            for i, neighbors in enumerate(neighbor_lists):
                n_neighbors = len(neighbors)
                neighbor_indices[i, :n_neighbors] = neighbors
                if n_neighbors < max_len:
                    neighbor_indices[i, n_neighbors:] = neighbors[-1]
            
            chunk_features = compute_geometric_features_batch(points, neighbor_indices, valid_mask)
            
            # Store results
            for feat_name, feat_values in chunk_features.items():
                all_features[feat_name][start_idx:end_idx] = feat_values
    
    return all_features


def adaptive_radius_simple(points: np.ndarray, base_radius: float = 0.01) -> float:
    """Simple density-based radius calculation."""
    n_points = len(points)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    volume = np.prod(maxs - mins)
    
    if volume <= 0:
        return base_radius
    
    density = n_points / volume
    density_threshold = 1.0 / (0.02 ** 3)
    
    if density >= density_threshold:
        return 0.03
    else:
        scale_factor = np.sqrt(density_threshold / density)
        return min(base_radius * scale_factor, 0.2)


def generate_geom_feat_from_pcd_faiss(params: Dict[str, Any], points_df: pd.DataFrame,
                                    output_dir: Path, input_file_stem: str) -> pd.DataFrame:
    """Fast geometric feature calculation using FAISS."""
    if not HAS_FAISS:
        raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
    
    base_radius = params["base_radius"]
    max_neighbors = params.get("max_neighbors", 50)
    export = params.get("export", True)
    out_signature_str = params.get("out_signature_str", "geom_feat")
    
    points_xyz = points_df[['X', 'Y', 'Z']].to_numpy()
    n_points = len(points_xyz)
    
    # Check GPU availability
    gpu_available = check_gpu_support()
    use_gpu = gpu_available and params.get("enable_gpu", True)
    
    print(f"Processing {n_points:,} points with FAISS {'GPU' if use_gpu else 'CPU'} algorithm")
    
    # Adaptive radius
    radius = adaptive_radius_simple(points_xyz, base_radius)
    print(f"Using radius: {radius:.4f}m")
    
    # Choose processing strategy
    if n_points > 1_000_000:  # Very large
        features = process_chunked_faiss(points_xyz, radius, max_neighbors, use_gpu=use_gpu)
    else:
        features = process_with_faiss(points_xyz, radius, max_neighbors, use_gpu=use_gpu)
    
    # Create result DataFrame
    result_df = points_df.copy()
    for feat_name, feat_values in features.items():
        result_df[feat_name] = feat_values
    
    # Clean NaN values
    feature_names = ['curvature', 'anisotropy', 'planarity', 'nx', 'ny', 'nz']
    for feat_name in feature_names:
        if feat_name in result_df.columns:
            valid_mask = np.isfinite(result_df[feat_name])
            if not valid_mask.all():
                result_df.loc[~valid_mask, feat_name] = 0.0
    
    # Export
    if export:
        save_dir = output_dir / 'pcd'
        create_dir_if_not_exists(save_dir)
        export_path = save_dir / f"{input_file_stem}_{out_signature_str}.txt"
        export_results(result_df, export_path)
        print(f"Results saved to {export_path}")
    
    return result_df


def main() -> None:
    """Main function for FAISS-based processing."""
    if not HAS_FAISS:
        print("❌ FAISS not available. Install with: pip install faiss-cpu")
        return
    
    start_time = time.time()
    
    current_config = get_config()
    if current_config is None:
        print("❌ Configuration not loaded")
        return
    
    global_params = current_config["global"]
    current_file_params = current_config["calc_geom_feature"]
    output_dir_ls = global_params["output_dir_ls"]
    input_path_ls = global_params["input_path_ls"]
    
    print(f"🚀 FAISS-based geometric feature calculation")
    print(f"Processing {len(input_path_ls)} files")
    
    for input_path, output_dir in zip(input_path_ls, output_dir_ls):
        print(f"\n⚡ Processing {input_path.stem}...")
        
        # Load data
        points_df = read_and_clean_pcd(
            input_path,
            cut_percent=current_file_params["cut_percent"],
            clean_pc=current_file_params["clean_pc"],
            dataset_name=global_params["dataset"],
            flip_mangrove=current_file_params["flip_mangrove"]
        )
        
        # Process
        file_start = time.time()
        result_df = generate_geom_feat_from_pcd_faiss(
            current_file_params, points_df, output_dir, input_path.stem
        )
        file_time = time.time() - file_start
        
        print(f"✅ Completed in {file_time:.1f}s ({len(result_df):,} points)")
        
        del points_df, result_df
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"\n🎉 Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
