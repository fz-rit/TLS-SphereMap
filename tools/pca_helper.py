from typing import Dict, Any
import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA, FastICA

def load_image_cube_and_metadata(image_cube_path: Path, metadata_path: Path) -> Dict[str, Any]:
    """
    Loads an image cube and its metadata from saved .npy files.

    Parameters:
    - image_cube_path: The path to the saved image cube file.
    - metadata_path: The path to the saved metadata file.

    Returns:
    - A dictionary containing the image cube and metadata.
    """
    
    # Load the image cube (8-channel data)
    image_cube = np.load(image_cube_path, allow_pickle=True)
    
    # Load the metadata from .json file
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    return image_cube, metadata


def compute_band_correlation(image):
    """
    Computes the correlation matrix between spectral bands of a multichannel image.

    Parameters:
    -----------
    image : np.ndarray
        A numpy array of shape (C, H, W) representing the multichannel image.
        C is the number of spectral bands.

    Returns:
    --------
    corr_matrix : np.ndarray
        A (C, C) correlation matrix between the spectral bands.
    """
    if image.ndim != 3:
        raise ValueError("Input image must have 3 dimensions (C, H, W)")

    C, H, W = image.shape
    reshaped = image.reshape(C, -1)  # Flatten spatial dimensions
    corr_matrix = np.corrcoef(reshaped)

    return corr_matrix

def compute_pca_components(image, n_components=3):
    """
    Applies PCA to a (C, H, W) image cube.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W).
    n_components : int
        Number of principal components to compute.

    Returns:
    --------
    pcs : np.ndarray
        Principal components reshaped to (n_components, H, W).
    pca : sklearn.decomposition.PCA
        The fitted PCA object (contains explained variance, components, etc.).
    """
    if image.ndim != 3:
        raise ValueError("Input image must be 3D (C, H, W)")

    C, H, W = image.shape
    reshaped = image.reshape(C, -1).T  # Shape: (H*W, C)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped)  # Shape: (H*W, n_components)
    pcs = pca_result.T.reshape(n_components, H, W)  # (n_components, H, W)

    return pcs, pca

def compute_mnf(image, noise_estimation=True, n_components=3):
    """
    Computes the Minimum Noise Fraction (MNF) transform.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W), where C is the number of bands.
    noise_estimation : bool
        If True, estimate noise as difference between adjacent pixels (simple method).
    n_components : int
        Number of MNF components to return.

    Returns:
    --------
    mnf_components : np.ndarray
        MNF components of shape (n_components, H, W).
    """
    C, H, W = image.shape
    X = image.reshape(C, -1).T  # Shape: (H*W, C)

    if noise_estimation:
        noise = X[1:] - X[:-1]
    else:
        noise = np.random.normal(0, 1, X.shape)

    noise_cov = np.cov(noise.T)
    signal_cov = np.cov(X.T)

    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(noise_cov) @ signal_cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    mnf_data = X @ eigvecs[:, :n_components]
    mnf_components = mnf_data.T.reshape(n_components, H, W)

    return mnf_components

def compute_ica(image, n_components=3):
    """
    Applies Independent Component Analysis (ICA) to a (C, H, W) image cube.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W).
    n_components : int
        Number of ICA components to compute.

    Returns:
    --------
    ica_components : np.ndarray
        ICA components of shape (n_components, H, W).
    """
    C, H, W = image.shape
    reshaped = image.reshape(C, -1).T  # Shape: (H*W, C)

    ica = FastICA(n_components=n_components, random_state=0)
    ica_result = ica.fit_transform(reshaped)
    ica_components = ica_result.T.reshape(n_components, H, W)

    return ica_components