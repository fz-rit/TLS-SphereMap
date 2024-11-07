import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def get_image_histogram(image_data: np.ndarray, output_dir:Path, title: str = '', saveflag: bool = False) -> None:
    """
    Generate and display a histogram of pixel values in the image data.

    Parameters:
    -----------
    image_data : np.ndarray
        A 2D array representing the density of points per pixel.
    title : str
        Title of the histogram plot.
    saveflag : bool, optional
        If True, the function will save the histogram plot to the output_dir directory (default is False).

    Returns:
    --------
    None
    """
    # Flatten the data to get a 1D array of values
    flattened_data = image_data.flatten()

    # Get unique values and their counts
    unique_values = np.unique(flattened_data)
    
    # Determine the number of bins
    if len(unique_values) > 20:
        bins = 20  # Limit bins to 20 if there are more than 20 unique values
        hist_values, bin_edges = np.histogram(flattened_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        normalized_counts = hist_values / hist_values.sum()  # Normalize counts
    else:
        values, counts = np.unique(flattened_data, return_counts=True)
        hist_values = counts
        bin_centers = values
        normalized_counts = hist_values / hist_values.sum()  # Normalize counts

    # Create the bar plot
    fig, ax1 = plt.subplots()

    # Absolute count bars
    ax1.bar(bin_centers, hist_values, width=0.8 * (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1,
            color='blue', alpha=0.7, label='Absolute Count')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Absolute Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add a second y-axis for normalized portion
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, normalized_counts, color='orange', marker='o', linestyle='-', label='Normalized Portion')
    ax2.set_ylabel('Normalized Portion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add grid, legend, and title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.show()
    print(f'Histogram:\nValues: {bin_centers}\nCounts: {hist_values}\nNormalized Counts: {normalized_counts.round(3)}')
    
    if saveflag:
        fig.savefig(f'{output_dir}/Histogram_{title}.png', dpi=300)
        print(f'Histogram saved to {output_dir} directory')


def get_histogram(input_vec: np.ndarray, output_dir:Path, title: str = '', saveflag: bool = False) -> None:
    """
    Generate and display a histogram of values in the input_vec data.

    Parameters:
    -----------
    image_data : np.ndarray
        A 1D array.
    output_dir : Path
        Path to the output directory.
    title : str
        Title of the histogram plot.
    saveflag : bool, optional
        If True, the function will save the histogram plot to the output_dir directory (default is False).

    Returns:
    --------
    None
    """
    # Get unique values and their counts
    unique_values = np.unique(input_vec)
    
    # Determine the number of bins
    if len(unique_values) > 20:
        bins = 20  # Limit bins to 20 if there are more than 20 unique values
        hist_values, bin_edges = np.histogram(input_vec, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        normalized_counts = hist_values / hist_values.sum()  # Normalize counts
    else:
        values, counts = np.unique(input_vec, return_counts=True)
        hist_values = counts
        bin_centers = values
        normalized_counts = hist_values / hist_values.sum()  # Normalize counts

    # Create the bar plot
    fig, ax1 = plt.subplots()

    # Absolute count bars
    ax1.bar(bin_centers, hist_values, width=0.8 * (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1,
            color='blue', alpha=0.7, label='Absolute Count')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Absolute Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add a second y-axis for normalized portion
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, normalized_counts, color='orange', marker='o', linestyle='-', label='Normalized Portion')
    ax2.set_ylabel('Normalized Portion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add grid, legend, and title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.show()
    print(f'Histogram:\nValues: {bin_centers}\nCounts: {hist_values}\nNormalized Counts: {normalized_counts.round(3)}')
    
    if saveflag:
        fig.savefig(f'{output_dir}/Histogram_{title}.png', dpi=300)
        print(f'Histogram saved to {output_dir} directory')