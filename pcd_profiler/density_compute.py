"""
Computation functions for point cloud density analysis.

This module contains functions for computing:
- Range from scan center
- Point density as a function of range
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_range_from_center(points, scan_center=None):
    """
    Compute Euclidean distance from each point to scan center.
    
    Parameters:
    -----------
    points : np.ndarray
        N x 3 array of point coordinates
    scan_center : np.ndarray or None
        3D coordinates of scan center. If None, uses centroid or origin.
        
    Returns:
    --------
    ranges : np.ndarray
        Distance of each point from scan center
    scan_center : np.ndarray
        The scan center used (returned for reference)
    """
    if scan_center is None:
        # Use centroid as default
        scan_center = np.mean(points, axis=0)
        print(f"Using point cloud centroid as scan center: {scan_center}")
    else:
        scan_center = np.array(scan_center)
        print(f"Using provided scan center: {scan_center}")
    
    # Compute Euclidean distance
    ranges = np.linalg.norm(points - scan_center, axis=1)
    
    return ranges, scan_center


def compute_density_vs_range(points, ranges, bin_width=1.0, density_type='volumetric', 
                             k_neighbors=None, merge_threshold=15.0):
    """
    Compute point density for range bins.
    
    Parameters:
    -----------
    points : np.ndarray
        N x 3 array of point coordinates
    ranges : np.ndarray
        Distance of each point from scan center
    bin_width : float
        Width of range bins in meters
    density_type : str
        'volumetric' for points/m³ or 'areal' for points/m²
    k_neighbors : int or None
        Number of neighbors for density estimation. If None, uses adaptive k.
    merge_threshold : float
        Range threshold (m) beyond which bins are merged into one
        
    Returns:
    --------
    bin_centers : np.ndarray
        Center of each range bin
    densities_median : np.ndarray
        Median density in each bin
    densities_q25 : np.ndarray
        25th percentile of density in each bin
    densities_q75 : np.ndarray
        75th percentile of density in each bin
    point_counts : np.ndarray
        Number of points in each bin
    """
    # Create bins up to merge threshold
    bins_list = list(np.arange(0, min(merge_threshold, np.max(ranges)) + bin_width, bin_width))
    
    # Add one large bin for everything beyond merge_threshold if needed
    if np.max(ranges) > merge_threshold:
        bins_list.append(np.max(ranges) + 0.01)
    
    bins = np.array(bins_list)
    
    # Compute bin centers
    bin_centers = []
    for i in range(len(bins) - 1):
        if bins[i] < merge_threshold:
            bin_centers.append(bins[i] + bin_width / 2)
        else:
            # For merged bin, use midpoint
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    bin_centers = np.array(bin_centers)
    
    # Assign points to bins
    bin_indices = np.digitize(ranges, bins) - 1
    
    densities_median = []
    densities_q25 = []
    densities_q75 = []
    point_counts = []
    
    print(f"\nComputing {density_type} density for {len(bins)-1} range bins...")
    if merge_threshold < np.max(ranges):
        print(f"Note: Bins beyond {merge_threshold}m are merged into one bin")
    
    for i in range(len(bins) - 1):
        # Get points in this bin
        mask = bin_indices == i
        bin_points = points[mask]
        count = np.sum(mask)
        point_counts.append(count)
        
        if count < 10:  # Skip bins with too few points
            densities_median.append(np.nan)
            densities_q25.append(np.nan)
            densities_q75.append(np.nan)
            continue
        
        # Compute local density using k-NN
        # Use adaptive k based on number of points, or use specified k
        if k_neighbors is None:
            k = min(50, count // 2)
        else:
            k = min(k_neighbors, count // 2)
        
        if k < 5:
            densities_median.append(np.nan)
            densities_q25.append(np.nan)
            densities_q75.append(np.nan)
            continue
        
        tree = cKDTree(bin_points)
        
        # For each point, find k nearest neighbors
        local_densities = []
        for point in bin_points[::max(1, count // 100)]:  # Sample points for efficiency
            distances, _ = tree.query(point, k=k+1)  # +1 to exclude the point itself
            distances = distances[1:]  # Remove the point itself
            
            if density_type == 'volumetric':
                # Volume of sphere containing k neighbors
                max_dist = distances[-1]
                if max_dist > 0:
                    volume = (4/3) * np.pi * max_dist**3
                    local_density = k / volume
                    local_densities.append(local_density)
            else:  # areal
                # Area of circle containing k neighbors (2D projection)
                max_dist = distances[-1]
                if max_dist > 0:
                    area = np.pi * max_dist**2
                    local_density = k / area
                    local_densities.append(local_density)
        
        if len(local_densities) > 0:
            densities_median.append(np.median(local_densities))
            densities_q25.append(np.percentile(local_densities, 25))
            densities_q75.append(np.percentile(local_densities, 75))
        else:
            densities_median.append(np.nan)
            densities_q25.append(np.nan)
            densities_q75.append(np.nan)
    
    return (bin_centers, 
            np.array(densities_median), 
            np.array(densities_q25),
            np.array(densities_q75),
            np.array(point_counts))


def compute_aggregate_density_vs_range(points_list, ranges_list, bin_width=1.0,
                                       density_type='volumetric', k_neighbors=None,
                                       merge_threshold=15.0):
    """
    Compute aggregated density statistics across multiple scans.
    
    For each scan, computes median density per bin independently.
    Then aggregates across scans by taking median and IQR of scan-level medians.
    
    Parameters:
    -----------
    points_list : list of np.ndarray
        List of N x 3 arrays, one per scan
    ranges_list : list of np.ndarray
        List of range arrays, one per scan
    bin_width : float
        Width of range bins in meters
    density_type : str
        'volumetric' for points/m³ or 'areal' for points/m²
    k_neighbors : int or None
        Number of neighbors for density estimation
    merge_threshold : float
        Range threshold (m) beyond which bins are merged
        
    Returns:
    --------
    bin_centers : np.ndarray
        Center of each range bin
    densities_median : np.ndarray
        Median of scan-level medians for each bin
    densities_q25 : np.ndarray
        25th percentile across scans
    densities_q75 : np.ndarray
        75th percentile across scans
    point_counts_median : np.ndarray
        Median point count across scans per bin
    scan_densities : list of np.ndarray
        Per-scan median densities for each bin (for plotting individual scans)
    scan_bin_centers : list of np.ndarray
        Per-scan bin centers (should be consistent)
    """
    n_scans = len(points_list)
    print(f"\nProcessing {n_scans} scans in aggregate mode...")
    print(f"{'='*60}")
    
    # Step 1: Compute per-scan statistics
    scan_densities = []
    scan_point_counts = []
    scan_bin_centers = []
    
    for i, (points, ranges) in enumerate(zip(points_list, ranges_list), 1):
        print(f"\nScan {i}/{n_scans}: {len(points)} points")
        bin_centers, dens_med, _, _, pt_counts = compute_density_vs_range(
            points, ranges, bin_width, density_type, k_neighbors, merge_threshold
        )
        scan_densities.append(dens_med)
        scan_point_counts.append(pt_counts)
        scan_bin_centers.append(bin_centers)
    
    # Use bin centers from first scan (should be consistent across scans)
    bin_centers = scan_bin_centers[0]
    n_bins = len(bin_centers)
    
    # Step 2: Aggregate across scans for each bin
    print(f"\n{'='*60}")
    print("Aggregating statistics across scans...")
    
    densities_median = []
    densities_q25 = []
    densities_q75 = []
    point_counts_median = []
    
    for bin_idx in range(n_bins):
        # Collect densities from all scans for this bin
        bin_densities = []
        bin_counts = []
        
        for scan_idx in range(n_scans):
            if bin_idx < len(scan_densities[scan_idx]):
                density = scan_densities[scan_idx][bin_idx]
                if not np.isnan(density):
                    bin_densities.append(density)
                bin_counts.append(scan_point_counts[scan_idx][bin_idx])
        
        # Compute aggregate statistics
        if len(bin_densities) > 0:
            densities_median.append(np.median(bin_densities))
            densities_q25.append(np.percentile(bin_densities, 25))
            densities_q75.append(np.percentile(bin_densities, 75))
        else:
            densities_median.append(np.nan)
            densities_q25.append(np.nan)
            densities_q75.append(np.nan)
        
        if len(bin_counts) > 0:
            point_counts_median.append(np.median(bin_counts))
        else:
            point_counts_median.append(0)
    
    return (bin_centers,
            np.array(densities_median),
            np.array(densities_q25),
            np.array(densities_q75),
            np.array(point_counts_median),
            scan_densities,
            scan_bin_centers)


def compute_vertical_density_distribution(points, bin_height=0.5, density_type='volumetric',
                                         k_neighbors=None):
    """
    Compute point density distribution by vertical (Z) bins.
    
    Parameters:
    -----------
    points : np.ndarray
        N x 3 array of point coordinates
    bin_height : float
        Height of vertical bins in meters
    density_type : str
        'volumetric' for points/m³ or 'areal' for points/m²
    k_neighbors : int or None
        Number of neighbors for density estimation. If None, uses adaptive k.
        
    Returns:
    --------
    bin_centers : np.ndarray
        Center of each vertical bin
    local_densities : list of np.ndarray
        Local density values for each point in each bin (for violin plots)
    bin_labels : np.ndarray
        Bin label for each local density value (for grouping violin plots)
    point_counts : np.ndarray
        Number of points in each bin
    """
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    
    # Create bins
    bins = np.arange(z_min, z_max + bin_height, bin_height)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Assign points to bins
    bin_indices = np.digitize(z_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    local_densities = []
    bin_labels = []
    point_counts = []
    
    print(f"\nComputing vertical {density_type} density for {len(bin_centers)} Z bins...")
    
    for i in range(len(bin_centers)):
        # Get points in this bin
        mask = bin_indices == i
        bin_points = points[mask]
        count = np.sum(mask)
        point_counts.append(count)
        
        if count < 10:  # Skip bins with too few points
            continue
        
        # Compute local density using k-NN
        if k_neighbors is None:
            k = min(50, count // 2)
        else:
            k = min(k_neighbors, count // 2)
        
        if k < 5:
            continue
        
        tree = cKDTree(bin_points)
        
        # For each point, compute local density
        for point in bin_points:
            distances, _ = tree.query(point, k=k+1)  # +1 to exclude the point itself
            distances = distances[1:]  # Remove the point itself
            
            if density_type == 'volumetric':
                # Volume of sphere containing k neighbors
                max_dist = distances[-1]
                if max_dist > 0:
                    volume = (4/3) * np.pi * max_dist**3
                    local_density = k / volume
                    local_densities.append(local_density)
                    bin_labels.append(f"{bin_centers[i]:.2f}m")
            else:  # areal
                # Area of circle containing k neighbors (2D projection)
                max_dist = distances[-1]
                if max_dist > 0:
                    area = np.pi * max_dist**2
                    local_density = k / area
                    local_densities.append(local_density)
                    bin_labels.append(f"{bin_centers[i]:.2f}m")
    
    return bin_centers, np.array(local_densities), np.array(bin_labels), np.array(point_counts)


def compute_vertical_density_distribution_aggregate(points_list, bin_height=0.5,
                                                   density_type='volumetric',
                                                   k_neighbors=None):
    """
    Compute vertical density distribution across multiple scans.
    Organizes data by scan ID (x-axis) with height distribution (y-axis).
    
    Parameters:
    -----------
    points_list : list of np.ndarray
        List of N x 3 arrays, one per scan
    bin_height : float
        Height of vertical bins in meters
    density_type : str
        'volumetric' for points/m³ or 'areal' for points/m²
    k_neighbors : int or None
        Number of neighbors for density estimation
        
    Returns:
    --------
    scan_heights : list of np.ndarray
        Heights (z-coordinates) for each scan
    scan_densities : list of np.ndarray
        Local densities corresponding to each height
    scan_ids : list of int
        Scan ID for each point (for grouping)
    z_range : tuple
        (z_min, z_max) across all scans
    """
    n_scans = len(points_list)
    print(f"\nProcessing {n_scans} scans for vertical density distribution...")
    
    scan_heights = []
    scan_densities = []
    scan_ids = []
    
    # Find common Z range across all scans
    z_min = min(points[:, 2].min() for points in points_list)
    z_max = max(points[:, 2].max() for points in points_list)
    z_range = (z_min, z_max)
    
    for scan_idx, points in enumerate(points_list, 1):
        print(f"Processing scan {scan_idx}/{n_scans} ({len(points):,} points)...")
        
        # Subsample for computational efficiency (take every Nth point)
        n_points = len(points)
        if n_points > 50000:
            sample_stride = n_points // 50000
            points_sampled = points[::sample_stride]
            print(f"  Subsampled to {len(points_sampled):,} points for efficiency")
        else:
            points_sampled = points
        
        # Compute local density for each point
        if k_neighbors is None:
            k = min(50, len(points_sampled) // 20)
        else:
            k = min(k_neighbors, len(points_sampled) // 20)
        
        if k < 5:
            print(f"  Skipping scan {scan_idx}: insufficient points")
            continue
        
        tree = cKDTree(points_sampled)
        
        local_densities = []
        heights = []
        
        # Compute density at each sampled point
        for point in points_sampled:
            distances, _ = tree.query(point, k=k+1)
            distances = distances[1:]  # Remove the point itself
            
            if density_type == 'volumetric':
                max_dist = distances[-1]
                if max_dist > 0:
                    volume = (4/3) * np.pi * max_dist**3
                    local_density = k / volume
                    local_densities.append(local_density)
                    heights.append(point[2])
            else:  # areal
                max_dist = distances[-1]
                if max_dist > 0:
                    area = np.pi * max_dist**2
                    local_density = k / area
                    local_densities.append(local_density)
                    heights.append(point[2])
        
        scan_heights.append(np.array(heights))
        scan_densities.append(np.array(local_densities))
        scan_ids.append([scan_idx] * len(heights))
        
        print(f"  Computed {len(heights):,} density values")
    
    return scan_heights, scan_densities, scan_ids, z_range
