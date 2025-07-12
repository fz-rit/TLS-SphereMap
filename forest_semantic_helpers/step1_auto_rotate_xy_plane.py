import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
import pandas as pd
import laspy
from forest_semantic_helpers.step0_read_pcd import read_pcd_file, observe_df


def compute_rotation_matrix(points: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align point cloud with Y-axis using minimum bounding box."""
    xy = points[:, :2] - np.mean(points[:, :2], axis=0)
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]

    min_area = float('inf')
    best_angle = 0

    for i in range(len(hull_pts)):
        edge = hull_pts[(i + 1) % len(hull_pts)] - hull_pts[i]
        angle = np.arctan2(edge[1], edge[0])
        
        R = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        rotated = (R @ hull_pts.T).T
        
        width = rotated[:, 0].max() - rotated[:, 0].min()
        height = rotated[:, 1].max() - rotated[:, 1].min()
        area = width * height

        if area < min_area:
            min_area = area
            best_angle = angle

    rotation_angle = (np.pi / 2) - best_angle
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    
    return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])


def apply_rotation(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply 2D rotation to 3D points preserving Z coordinates."""
    points_homo = np.hstack([points[:, :2], np.ones((points.shape[0], 1))])
    rotated_xy = (rotation_matrix @ points_homo.T).T[:, :2]
    return np.hstack([rotated_xy, points[:, 2:3]])


def rotate_original_points_with_fewer_ground_points(original_df, sample_ground_df, rotation_matrix):
    """
    Replace original ground points with fewer sampled ground points and apply rotation.
    
    Parameters:
        original_df (pd.DataFrame): Original point cloud DataFrame
        sample_ground_df (pd.DataFrame): Sampled ground points DataFrame  
        rotation_matrix (np.ndarray): 3x3 rotation matrix
        
    Returns:
        pd.DataFrame: Updated DataFrame with fewer ground points and rotation applied
    """
    # Remove original ground points (Classification == 1)
    non_ground_df = original_df[original_df['Classification'] != 1].copy()
    print(f"Non-ground points: {len(non_ground_df):,}")
    print(f"Sampled ground points: {len(sample_ground_df):,}")
    
    # Combine non-ground points with sampled ground points
    updated_df = pd.concat([non_ground_df, sample_ground_df], ignore_index=True)
    print(f"Combined points: {len(updated_df):,}")
    
    # Apply rotation to all points
    points = updated_df[['X', 'Y', 'Z']].values
    rotated_points = apply_rotation(points, rotation_matrix)
    updated_df[['X', 'Y', 'Z']] = rotated_points
    
    print(f"Applied rotation to {len(updated_df):,} total points")
    return updated_df


def main():
    """Main processing function."""
    # Configuration
    pcd_path = Path("/home/fzhcis/Downloads/ForestSemantic/Plot_1.las")
    output_dir = pcd_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pcd_path.stem}_ground_sample_points.csv"
    
    # Load and filter data
    print(f"Loading {pcd_path.name}...")
    pcd_df = read_pcd_file(pcd_path, verbose=True)
    print(f"Total points: {len(pcd_df):,}")

    # Filter ground points and subsample
    ground_df = pcd_df[pcd_df['Classification'] == 1].copy()
    ground_sample_df = ground_df.sample(n=int(len(ground_df) * 0.01), random_state=42).reset_index(drop=True)
    print(f"\n=====Ground points: {len(ground_df):,}, Sampled: {len(ground_sample_df):,}======")
    
    # Compute rotation matrix from original sample points
    ground_sample_pts = ground_sample_df[['X', 'Y', 'Z']].values
    rotation_matrix = compute_rotation_matrix(ground_sample_pts)
    
    # Apply rotation to sample points for saving
    rotated_sample_points = apply_rotation(ground_sample_pts, rotation_matrix)
    ground_sample_df_rotated = ground_sample_df.copy()
    ground_sample_df_rotated[['X', 'Y', 'Z']] = rotated_sample_points
    
    # Save rotated sample points
    observe_df(ground_sample_df_rotated)
    ground_sample_df_rotated.to_csv(output_path, index=False)
    
    # Apply rotation to full dataset with original (unrotated) sampled ground points
    print(f"\n=====Processing full dataset with sampled ground points======")
    updated_df = rotate_original_points_with_fewer_ground_points(pcd_df, ground_sample_df, rotation_matrix)
    updated_output_path = output_dir / f"{pcd_path.stem}_rotated_with_sampled_ground.csv"
    updated_df.to_csv(updated_output_path, index=False)
    
    # Save rotation matrix
    np.savetxt(output_dir / "align_Y_axis_rotation_matrix.txt", rotation_matrix)

    print(f"Ground sample points saved to: {output_path}")
    print(f"Full rotated dataset saved to: {updated_output_path}")
    print("Processing completed!")


if __name__ == "__main__":
    main()