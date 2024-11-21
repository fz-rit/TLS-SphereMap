import numpy as np
from typing import Tuple

def extrinsic_mtx_face_to_origin(lidar_position: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate the extrinsic matrix for a LIDAR at a given position facing the origin (0, 0, 0).
    
    Parameters:
        lidar_position (Tuple[float, float, float]): The position of the LIDAR in world coordinates, e.g., (x, y, z).
    
    Returns:
        np.ndarray: The 4x4 extrinsic transformation matrix.
    """
    # Convert lidar position to a numpy array
    lidar_position = np.array(lidar_position)
    
    # Calculate the forward vector (pointing from lidar to origin)
    forward_vector = -lidar_position
    forward_vector = forward_vector / np.linalg.norm(forward_vector)  # Normalize
    
    # Define an arbitrary up vector (typically along the Z-axis for simplicity)
    up_vector = np.array([0, 0, 1])
    
    # Calculate the right vector (cross product of up and forward vectors)
    right_vector = np.cross(up_vector, forward_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)  # Normalize
    
    # Recalculate the up vector to ensure orthogonality (cross product of forward and right)
    up_vector = np.cross(forward_vector, right_vector)
    
    # Construct the rotation matrix using the right, up, and forward vectors as columns
    R = np.column_stack((right_vector, up_vector, forward_vector))
    
    # Construct the extrinsic matrix as a 4x4 matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = lidar_position
    
    return extrinsic_matrix

if __name__ == "__main__":
    # Example usage
    lidar_position = (3, -3, 3)
    extrinsic_matrix = extrinsic_mtx_face_to_origin(lidar_position)
    print("Extrinsic Matrix:\n", extrinsic_matrix)
