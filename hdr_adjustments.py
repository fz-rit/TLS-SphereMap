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

def contrast_enhancement(image, method='clahe', **kwargs):
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
    print(f"Image value range before adjustment: [{image.min()}, {image.max()}]")
    image = exposure.rescale_intensity(image, out_range=(0, 1))
    print("Image rescaled to [0, 1].")

    stretch_percent = kwargs.get('stretch_percent', 2)
    image = percentile_stretching(image, stretch_percent)
    print(f"Stretching image by {stretch_percent} percent.")

    if method == 'clahe':
        adjusted_image = exposure.equalize_adapthist(image)
        print("CLAHE applied.")

    elif method == 'log':
        c = kwargs.get('scale_factor', 1)
        adjusted_image = c * np.log(1 + image)

    elif method == 'piecewise':
        breakpoints = kwargs.get('breakpoints', [(0, 0.7, 5), (0.7, 0.8, 2), (0.8, 1.0, 0.5)])
        adjusted_image = np.zeros_like(image)
        for start, end, slope in breakpoints:
            mask = (image >= start) & (image <= end)
            adjusted_image[mask] = slope * (image[mask] - start)

    elif method == 'sigmoid':
        cutoff = kwargs.get('cutoff', 0.05)
        gain = kwargs.get('gain', 2)
        adjusted_image = exposure.adjust_sigmoid(image, cutoff=cutoff, gain=gain)

    elif method == 'weighting':
        alpha = kwargs.get('alpha', 0.02)
        adjusted_image = np.exp(-alpha * image) * image


    # contrast enhancement from opencv package
    # elif method == 'contrast_stretching':
    #     p_low, p_high = np.percentile(image, (2, 98))
    #     adjusted_image = exposure.rescale_intensity(image, in_range=(p_low, p_high))


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


