from typing import Dict, Any
import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA, FastICA


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
    image = image.transpose(2, 0, 1) # Change (H,W,C) to (C, H, W)
    C, H, W = image.shape
    reshaped = image.reshape(C, -1)  # Flatten spatial dimensions
    corr_matrix = np.corrcoef(reshaped)

    return corr_matrix

def compute_pca_components(image, n_components=3):
    """
    Applies PCA to a (H, W, C) image cube.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (H, W, C).
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
        raise ValueError("Input image must be 3D (H, W, C)")

    H, W, C = image.shape
    reshaped = image.reshape(-1, image.shape[2])  # Shape: (H*W, C)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped)  # Shape: (H*W, n_components)
    pcs = pca_result.reshape(H, W, n_components)

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
    H, W, C = image.shape
    reshaped = image.reshape(-1, image.shape[2])  # Shape: (H*W, C)

    if noise_estimation:
        noise = reshaped[1:] - reshaped[:-1]
    else:
        noise = np.random.normal(0, 1, reshaped.shape)

    noise_cov = np.cov(noise.T)
    signal_cov = np.cov(reshaped.T)

    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(noise_cov) @ signal_cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    mnf_data = reshaped @ eigvecs[:, :n_components]
    mnf_components = mnf_data.reshape(H, W, n_components)

    return mnf_components # shape (H, W, n_components)

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
    H, W, C = image.shape
    reshaped = image.reshape(-1, image.shape[2])  # Shape: (H*W, C)

    ica = FastICA(n_components=n_components, random_state=0)
    ica_result = ica.fit_transform(reshaped)
    ica_components = ica_result.reshape(H, W, n_components)

    return ica_components # shape (H, W, n_components)