import open3d as o3d
import numpy as np
from pathlib import Path
from preprocess_point_cloud import preprocess_point_cloud
from tqdm import tqdm
from plot_tools import get_histogram
import torch

def pad_neighbors(points, neighbors_list, max_neighbors):
    """ Pad each neighborhood to the maximum number of neighbors for batch processing """
    padded_neighbors = []
    mask = []
    
    for idx in neighbors_list:
        neighbors = points[idx]  # Get neighbors for each point
        if len(neighbors) < max_neighbors:
            # Pad with zeros if neighborhood is smaller than max_neighbors
            padded = torch.cat([neighbors, torch.zeros((max_neighbors - len(neighbors), 3), device=points.device)])
            mask.append(torch.cat([torch.ones(len(neighbors), device=points.device), torch.zeros(max_neighbors - len(neighbors), device=points.device)]))
        else:
            # Truncate if neighborhood is larger than max_neighbors
            padded = neighbors[:max_neighbors]
            mask.append(torch.ones(max_neighbors, device=points.device))
        
        padded_neighbors.append(padded)
    
    return torch.stack(padded_neighbors), torch.stack(mask)

def estimate_curvature_batched(pcd, radius=0.05, max_neighbors=10):
    # Convert Open3D point cloud to a PyTorch tensor
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    
    # KDTree for fast neighborhood search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    neighbors_list = []
    
    # Collect all neighbors in a single loop
    for point in tqdm(points.cpu().numpy(), desc="Finding Neighbors"):
        _, idx, _ = pcd_tree.search_radius_vector_3d(point, radius)
        neighbors_list.append(idx)
    
    # Pad and stack neighborhoods for batched processing
    padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)
    
    # Center neighbors for covariance calculation
    centroids = padded_neighbors.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # Calculate centroids
    centered_neighbors = padded_neighbors - centroids.unsqueeze(1)  # Centered neighborhoods
    
    # Calculate covariance matrices in a batched way
    covariances = torch.matmul(centered_neighbors.transpose(1, 2), centered_neighbors * mask.unsqueeze(2)) / (mask.sum(dim=1) - 1).view(-1, 1, 1)
    
    # Perform eigen decomposition in a batched way
    eigenvalues, _ = torch.linalg.eigh(covariances)  # Eigenvalues are sorted in ascending order
    
    # Curvature is the ratio of the smallest eigenvalue to the sum of all eigenvalues
    curvatures = eigenvalues[:, 0] / eigenvalues.sum(dim=1)
    
    return curvatures.cpu().numpy()

# def estimate_roughness(pcd, radius=0.1):
#     # Convert Open3D point cloud to numpy array for convenience
#     points = np.asarray(pcd.points)
    
#     # Placeholder for roughness values
#     roughness = np.zeros(points.shape[0])
    
#     # KDTree for fast neighborhood search
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
#     # Iterate with tqdm for progress display
#     for i, point in tqdm(enumerate(points), total=len(points), desc="Calculating Roughness"):
#         # Search for neighboring points within the specified radius
#         _, idx, _ = pcd_tree.search_radius_vector_3d(point, radius)
        
#         if len(idx) < 3:
#             continue  # Skip if there aren't enough neighbors for a plane fit
        
#         # Get the neighboring points
#         neighbors = points[idx, :]
        
#         # Calculate the centroid of the neighbors
#         centroid = np.mean(neighbors, axis=0)
        
#         # Center the neighbors around the centroid
#         centered_neighbors = neighbors - centroid
        
#         # Calculate covariance matrix and eigenvalues for PCA
#         covariance_matrix = np.cov(centered_neighbors.T)
#         eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
#         # Get the normal of the plane (eigenvector with smallest eigenvalue)
#         normal = eigenvectors[:, np.argmin(eigenvalues)]
        
#         # Calculate roughness as average distance from each neighbor to the plane
#         distances = np.dot(centered_neighbors, normal)  # Project onto normal
#         roughness[i] = np.mean(np.abs(distances))
    
#     return roughness


# Example usage
# Load your point cloud
# Create a PointCloud object and assign the points

filename = Path(r'C:\Users\fzhcis\Documents\projects\from_RobC\for_Fei\data\palau_2024\ALRSET1\UMBCBL009_2024-03-28-02-47-26_ALRSET12_060180_000200.800_1830507489.txt')
output_dir = Path(r'C:\Users\fzhcis\Documents\mylab\tls_demo\outputs\1106\outputs')
df_filtered = preprocess_point_cloud(filename)
points = df_filtered[['X', 'Y', 'Z']].to_numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate curvature
curvatures = estimate_curvature_batched(pcd, radius=0.1)
get_histogram(curvatures, output_dir, title='Curvature', saveflag=False)
# You can add curvatures to the point cloud as colors for visualization
pcd.colors = o3d.utility.Vector3dVector(np.tile(curvatures[:, None], (1, 3)))
o3d.visualization.draw_geometries([pcd])
