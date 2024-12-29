"""
Contributor: fzhcis@rit.edu
Version: 1.0
Last Updated: 11/19/2024
Description:
This module contains functions for generating and displaying histograms of image data and input vectors, 
as well as displaying unwrapped images with titles and colorbars.
Functions:
----------
- get_image_histogram(image_data: np.ndarray, output_dir: Path, title: str = '', saveflag: bool = False) -> None:
- get_histogram(input_vec: np.ndarray, output_dir: Path, title: str = '', saveflag: bool = False) -> None:
    Generate and display a histogram of values in the input vector data.
- display_unwrapped_images(subplot_images: tuple[np.ndarray], titles: tuple[str], output_dir: Path, 
    colormap: str = 'plasma', saveflag: bool = False) -> None:
    Display images with titles and colorbars.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from skimage import io
from matplotlib.colors import ListedColormap, BoundaryNorm

AZIMUTH_NORM_SCALE = 360
ZENITH_NORM_SCALE = 135
# Since degree resolution=0.25: 360/0.25=1440
CANVAS_WIDTH = 1440 
CANVAS_HEIGHT = 540

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


def get_histogram(input_vec: np.ndarray, 
                  output_dir:Path, 
                  title: str = '', 
                  saveflag: bool = False,
                  log_y: bool = False) -> None:
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

    if log_y:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    # Add grid, legend, and title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.show()
    print(f'Histogram:\nValues: {bin_centers}\nCounts: {hist_values}\nNormalized Counts: {normalized_counts.round(3)}')
    
    if saveflag:
        fig.savefig(f'{output_dir}/Histogram_{title}.png', dpi=300)
        print(f'Histogram saved to {output_dir} directory')


def display_unwrapped_single_band_images(subplot_images: tuple[np.ndarray], 
                             titles: tuple[str],
                             output_dir: Path,
                             colormap: str = 'plasma', 
                             saveflag: bool = False,
                             upside_down: bool = False) -> None:
    """
    Display images with titles and colorbars.

    Parameters:
    subplot_images (np.ndarray): A tuple of images to display.
    titles (str): Titles for the images.
    colormap (str): Colormap to use for displaying the images. Default is 'plasma'.
    saveflag (bool): If True, save the images to the output_dir. Default is False.

    Returns:
    None
    """

    num_subplots = len(subplot_images)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4*num_subplots))

    # Set the colormap to something more visually friendly
    colormap = colormap  # e.g.: 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # Define custom ticks using np.linspace
    y_ticks = np.linspace(0, ZENITH_NORM_SCALE, CANVAS_HEIGHT + 1) if upside_down else 135 - np.linspace(0, ZENITH_NORM_SCALE, CANVAS_HEIGHT + 1)
    x_ticks = np.linspace(0, AZIMUTH_NORM_SCALE, CANVAS_WIDTH + 1)
    y_label = 'Elevation Angle (degree)' if upside_down else 'Zenith Angle (degree)'
    for ax, subplot_img, title in zip(axes, subplot_images, titles):
        im = ax.imshow(subplot_img, cmap=colormap, aspect='auto', extent=[x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]])
        ax.set_xlabel('Azimuth Angle (degrees)')
        ax.set_ylabel(y_label)
        ax.set_xticks(x_ticks[::int(len(x_ticks) / 10)])  # Reduce the number of x-ticks to avoid overlap
        ax.set_yticks(y_ticks[::int(len(y_ticks) / 10)])  # Reduce the number of y-ticks for readability
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Adjust colorbar size
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

    if saveflag:
        for i, title in enumerate(titles):
            if title == 'Intensity Map (adjusted)':
                plt.imsave(output_dir / f'{title}.png', subplot_images[i], cmap=colormap)
        
        fig.savefig(output_dir / 'combined_unwrapped_images.png')
        print(f'Images saved to {output_dir} directory')




def display_single_band_img_wt_discrete_values(
    image_data: np.ndarray,
    output_dir: Path,
    title: str = "Point Density Map",
    saveflag: bool = False
) -> None:
    """
    Displays a single-band image with discrete values using a custom colormap, and optionally saves the image.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to display, expected to contain discrete integer values.
    output_dir : Path
        The directory where the image will be saved if `saveflag` is True.
    title : str, optional
        The title of the image, by default "Point Density Map".
    saveflag : bool, optional
        If True, saves the image to the output directory, by default False.

    Returns
    -------
    None
        This function does not return any value.
    """
    # Display the image with the discrete colormap
    fig, ax = plt.subplots(figsize=(18, 5))
    unique_values = np.unique(image_data)
    num_unique_values = len(unique_values)

    if num_unique_values < 50:
        colors = plt.get_cmap('jet', num_unique_values)(np.arange(num_unique_values))
    else:
        # Sample the 'jet' colormap to get 50 colors
        jet_colors = plt.get_cmap('jet', 50)(np.linspace(0, 1, 50))
        # Repeat the colors to match the number of unique values
        repeated_colors = np.tile(jet_colors, (int(np.ceil(num_unique_values / 50)), 1))[:num_unique_values]
        # Shuffle the colors to make adjacent values more distinguishable
        np.random.seed(1)  # For reproducibility
        np.random.shuffle(repeated_colors)
        colors = repeated_colors

    colors[0] = [0, 0, 0, 1]  # Set the color for zero to black (RGBA)
    cmap = ListedColormap(colors)

    # Create a boundary norm with explicit bounds
    boundaries = np.concatenate([[unique_values[0] - 0.5], unique_values + 0.5])
    norm = BoundaryNorm(boundaries, num_unique_values)

    im = ax.imshow(image_data, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, ticks=unique_values)
    cbar.set_ticklabels([str(int(i)) for i in unique_values])
    cbar.set_label('Points per pixel')
    ax.set_title(title)
    plt.show()


    if saveflag:
        # Save the raw image without labels or colorbars
        normalized_data = norm(image_data)
        rgba_image = cmap(normalized_data)
        plt.imsave(output_dir / f'{title}.png', rgba_image, format='png', dpi=1)

        # # Save the displayed image with annotations
        # fig.savefig(output_dir / f'{title}.png', dpi=600)
        print(f'Images saved to {output_dir} directory')


def display_unwrapped_rgb_image(rgb_image: np.ndarray, 
                            figure_title: str, 
                            output_dir: Path, 
                            saveflag: bool = False,
                            upside_down: bool = False,
                            ) -> None:
    """
    Display the unwrapped RGB image with custom ticks.
    """
    y_ticks = np.linspace(0, ZENITH_NORM_SCALE, CANVAS_HEIGHT + 1) if upside_down else 135 - np.linspace(0, ZENITH_NORM_SCALE, CANVAS_HEIGHT + 1)
    x_ticks = np.linspace(0, AZIMUTH_NORM_SCALE, CANVAS_WIDTH + 1)
    y_label = 'Elevation Angle (degree)' if upside_down else 'Zenith Angle (degree)'
    plt.figure(figsize=(12, 6))
    plt.imshow(rgb_image, aspect='auto', extent=[x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]])
    
    plt.title(figure_title)
    plt.xticks(x_ticks[::int(len(x_ticks) / 10)])  # Reduce the number of x-ticks to avoid overlap
    plt.yticks(y_ticks[::int(len(y_ticks) / 10)])  # Reduce the number of y-ticks for readability
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel(y_label)
    plt.show()

    if saveflag:
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        io.imsave(f'{output_dir}/{figure_title}.tif', rgb_image_uint8)