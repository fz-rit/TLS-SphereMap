import numpy as np
from skimage import exposure

def hdr_adjustment(image, method='clahe', **kwargs):
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
    # Normalize the image by dividing by the maximum value and ensure values are in range [0, 1]
    image = image.astype(np.float32)
    image = image / np.max(image)
    image = np.clip(image, 0, 1)

    if method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 0.03)
        adjusted_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)

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
        cutoff = kwargs.get('cutoff', 0.5)
        gain = kwargs.get('gain', 10)
        adjusted_image = exposure.adjust_sigmoid(image, cutoff=cutoff, gain=gain)

    elif method == 'weighting':
        alpha = kwargs.get('alpha', 0.02)
        adjusted_image = np.exp(-alpha * image) * image

    else:
        raise ValueError("Invalid method specified. Choose from 'clahe', 'log', 'piecewise', 'sigmoid', 'weighting'.")

    return adjusted_image


