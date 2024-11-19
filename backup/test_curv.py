import torch

batch_size = 10000
max_neighbors = 20
points = torch.randn(batch_size, 3, device='cuda')

# Dummy data for testing
def batch_neighborhood_search(points, radius, max_neighbors):
    # For testing, create random neighbors
    neighbors_list = [torch.randperm(batch_size)[:max_neighbors].tolist() for _ in range(batch_size)]
    return neighbors_list

def pad_neighbors(points, neighbors_list, max_neighbors):
    num_points = points.shape[0]
    padded_neighbors = torch.zeros((num_points, max_neighbors, 3), device=points.device)
    mask = torch.zeros((num_points, max_neighbors), device=points.device)

    for i, idx in enumerate(neighbors_list):
        num_neighbors = len(idx)
        if num_neighbors > max_neighbors:
            idx = idx[:max_neighbors]
            num_neighbors = max_neighbors
        padded_neighbors[i, :num_neighbors] = points[idx]
        mask[i, :num_neighbors] = 1

    return padded_neighbors, mask


def estimate_curvature_roughness_batched(points, curvature_radius=0.05, roughness_radius=0.2, max_neighbors=10):
    """Estimate curvature and roughness for a batch of points with GPU-based neighborhood search."""
    points = torch.tensor(points, dtype=torch.float32, device='cuda')

    def compute_covariances(points, radius):
        neighbors_list = batch_neighborhood_search(points, radius, max_neighbors)
        padded_neighbors, mask = pad_neighbors(points, neighbors_list, max_neighbors)

        # Compute centroids
        mask_sum = mask.sum(dim=1, keepdim=True).unsqueeze(2)
        centroids = (padded_neighbors * mask.unsqueeze(2)).sum(dim=1, keepdim=True) / mask_sum

        # Centered neighbors
        centered_neighbors = padded_neighbors - centroids

        # Apply mask to centered_neighbors
        centered_neighbors_masked = centered_neighbors * mask.unsqueeze(2)
        print("centered_neighbors shape:", centered_neighbors.shape)
        print("mask.unsqueeze(2) shape:", mask.unsqueeze(2).shape)
        print("centered_neighbors_masked shape:", centered_neighbors_masked.shape)
        print("centered_neighbors_masked.transpose(1, 2) shape:", centered_neighbors_masked.transpose(1, 2).shape)
        print("mask_sum shape:", mask_sum.shape)
        print("mask_sum.squeeze(2) shape:", mask_sum.squeeze(2).shape)
        print("mask_sum.squeeze(2)-1 shape:", (mask_sum.squeeze(2)-1).shape)
        # Compute covariance matrices using torch.bmm
        cov_tensor_a = torch.bmm(centered_neighbors_masked.transpose(1, 2), centered_neighbors_masked)
        cov_tensor_b = mask_sum.squeeze(2) - 1
        print("cov_tensor_a shape:", cov_tensor_a.shape)
        print("cov_tensor_b shape:", cov_tensor_b.shape)
        denom = torch.clamp(mask_sum.squeeze(2) - 1, min=1) # Clamp to ensure denominator is at least 1
        covariances = cov_tensor_a / denom.unsqueeze(1).unsqueeze(2)  
        # covariances = torch.bmm(centered_neighbors_masked.transpose(1, 2), centered_neighbors_masked) / (mask_sum.squeeze(2) - 1)
        # covariances = cov_tensor_a / cov_tensor_b.unsqueeze(1).unsqueeze(2)

        return covariances, mask, centered_neighbors_masked

    # Calculate curvature
    covariances, _, _ = compute_covariances(points, curvature_radius)

    # Compute eigenvalues for curvature
    eigenvalues, _ = torch.linalg.eigh(covariances)

    # Curvature calculation
    curvatures = eigenvalues[:, 0] / eigenvalues.sum(dim=1)

    # Calculate roughness
    covariances, mask, centered_neighbors_masked = compute_covariances(points, roughness_radius)

    # Compute eigenvalues and eigenvectors for roughness
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)

    # Normal vectors (corresponding to smallest eigenvalue)
    normal_vectors = eigenvectors[:, :, 0]

    # Roughness calculation
    roughness_numerators = torch.abs(torch.bmm(centered_neighbors_masked, normal_vectors.unsqueeze(2)).squeeze())
    roughness = roughness_numerators.sum(dim=1) / mask.sum(dim=1)

    return curvatures.cpu().numpy(), roughness.cpu().numpy()


# Use the updated function
curvatures, roughness = estimate_curvature_roughness_batched(points.cpu().numpy())
print("Curvatures shape:", curvatures.shape)
print("Roughness shape:", roughness.shape)

