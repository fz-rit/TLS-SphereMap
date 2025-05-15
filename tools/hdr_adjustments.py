import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


def percentile_stretching(image: np.ndarray, percent: int=1) -> np.ndarray:
    """
    Perform percentile stretching on the given image.

    Args:
        image: Image to stretch.
        low: Lower percentile.
        high: Higher percentile.

    Returns: Stretched image.
    """
    assert 0 <= percent <= 10, f"Percentile must be between 0 and 10, but got {percent}"
    p_low, p_high = np.percentile(image, (percent, 100-percent))
    return exposure.rescale_intensity(image, in_range=(p_low, p_high))

def contrast_enhancement(image, method='hist_equal', **kwargs):
    """
    Apply HDR dynamic range adjustment to an image using a specified method.

    Parameters:
    - image (ndarray): The input image to be processed.
    - method (str): The method to use for HDR adjustment. Options are:
        'clahe' - Contrast Limited Adaptive Histogram Equalization
        'log' - Logarithmic Transformation
        'piecewise' - Piecewise Linear Transformation
        'sigmoid' - Sigmoid Stretching
        'weighting' - Custom Weighting Function
    - kwargs: Additional parameters for specific methods.

    Returns:
    - adjusted_image (ndarray): The image after applying the specified HDR adjustment.
    """
    # Assuming image has been normalized to [0, 1]
    image = image.astype(np.float32)

    # Create a mask for the void pixels, with value 0
    zero_mask = image == 0


    stretch_percent = kwargs.get('stretch_percentile', 1)
    if stretch_percent > 0:
        image[~zero_mask] = percentile_stretching(image[~zero_mask], stretch_percent)
        print(f"Stretching image by {stretch_percent} percent. (Void pixels excluded)")

    if method == 'clahe':
        adjusted_image = exposure.equalize_adapthist(image)
        print("CLAHE applied.")

    elif method == 'hist_equal':
        adjusted_image = exposure.equalize_hist(image, mask=~zero_mask)
        channel_info = kwargs.get('channel_info', None)
        if channel_info is not None:
            print(f"{channel_info} - Global Histogram Equalization applied. (Void pixels excluded)")

    elif method == 'log':
        c = kwargs.get('scale_factor', 1)
        adjusted_image = image.copy()
        adjusted_image[~zero_mask] = c * np.log(1 + image[~zero_mask])
        print("Logarithmic Transformation applied. (Void pixels excluded)")

    else:
        raise ValueError("Invalid method specified. Choose from 'clahe', 'log', 'piecewise', 'sigmoid', 'weighting'.")
    
    compare_hist = kwargs.get('compare_hist', False)
    if compare_hist:
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        ax = axes.ravel()
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].hist(image.ravel(), bins=256, histtype='step', color='black')
        ax[1].set_title('Original Histogram')
        ax[2].imshow(adjusted_image, cmap='gray')
        ax[2].set_title('Adjusted Image')
        ax[3].hist(adjusted_image.ravel(), bins=256, histtype='step', color='black')
        ax[3].set_title('Adjusted Histogram')
        plt.tight_layout()
        plt.show()

    return adjusted_image


