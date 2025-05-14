"""
Contributor: fzhcis@rit.edu
Version: 1.0
Last Updated: 04/29/2025
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
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from PIL import Image

# HORIZONTAL_FOV = 360.0
# # ======For TLS data======
# VERTICAL_FOV = 135.0
# VERTICAL_ANGLE_RESOLUTION = 0.25
# HORIZONTAL_ANGLE_RESOLUTION = 0.25

# # # ===For SemanticKitti data (Velodyne-HDL-64)===
# # VERTICAL_FOV = 30.0
# # VERTICAL_ANGLE_RESOLUTION = 0.4
# # HORIZONTAL_ANGLE_RESOLUTION = 0.08
# CANVAS_WIDTH = int(HORIZONTAL_FOV / HORIZONTAL_ANGLE_RESOLUTION) # 1440 for TLS data; 4500 for SemanticKitti data
# CANVAS_HEIGHT = int(VERTICAL_FOV / VERTICAL_ANGLE_RESOLUTION) # 540 for TLS data; 75 for SemanticKitti data



def get_image_histogram(image_data: np.ndarray, 
                        output_dir: Path, 
                        title: str = '', 
                        saveflag: bool = False,
                        visualize: bool = True
                        ) -> None:
    """
    Generate and display a histogram of pixel values in the image data,
    distributing bars evenly along the x-axis based on unique pixel values.

    Parameters:
    -----------
    image_data : np.ndarray
        A 2D array representing the density of points per pixel.
    output_dir : Path
        Path to the output directory for saving the histogram (if saveflag=True).
    title : str
        Title of the histogram plot.
    saveflag : bool, optional
        If True, the function saves the histogram plot to the output_dir directory (default is False).

    Returns:
    --------
    None
    """
    # Input validation
    if not isinstance(image_data, np.ndarray):
        raise TypeError("image_data must be a numpy array.")
    if image_data.ndim != 2:
        raise ValueError("image_data must be a 2D array.")
    if saveflag and not output_dir.is_dir():
        raise FileNotFoundError(f"The output directory {output_dir} does not exist.")

    # Flatten the data to get a 1D array of values
    flattened_data = image_data.flatten()

    # Get unique values and their counts
    values, counts = np.unique(flattened_data, return_counts=True)
    normalized_counts = counts / counts.sum()  # Normalize counts

    # Map values to evenly spaced indices for plotting
    x_indices = np.arange(len(values))

    # Create the bar plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Absolute count bars
    ax1.bar(x_indices, counts, color='blue', alpha=0.7, label='Absolute Count')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Absolute Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add a second y-axis for normalized portion
    ax2 = ax1.twinx()
    ax2.plot(x_indices, normalized_counts, color='orange', marker='o', linestyle='-', label='Normalized Portion')
    ax2.set_ylabel('Normalized Portion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Replace x-ticks with the actual pixel values
    plt.xticks(x_indices, labels=[f"{int(v)}" if v < 100 else f"{int(v):,}" for v in values], rotation=45)

    # Add grid, legend, and title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Adjust layout for better spacing
    fig.tight_layout()
    

    # Print histogram data
    print(f'Histogram:\nValues: {values}\nCounts: {counts}\nNormalized Counts: {normalized_counts.round(3)}')

    # Save the figure if saveflag is True
    if saveflag:
        save_path = output_dir / f'Histogram_{title}.png'
        fig.savefig(save_path, dpi=300)
        print(f'Histogram saved to {save_path}')
    
    if visualize:
        plt.show()


def get_vector_histogram(input_vec: np.ndarray, 
                         output_dir: Path, 
                         title: str = '', 
                         saveflag: bool = True,
                         log_y: bool = True,
                         visualize: bool = True
                         ) -> None:
    """
    Generate and display a histogram of values in the input_vec data.

    Parameters:
    -----------
    input_vec : np.ndarray
        A 1D array of numerical data for generating the histogram.
    output_dir : Path
        Path to the output directory for saving the histogram (if saveflag=True).
    title : str
        Title of the histogram plot.
    saveflag : bool, optional
        If True, the function saves the histogram plot to the output_dir directory (default is False).
    log_y : bool, optional
        If True, the Y-axes (absolute and normalized) will use logarithmic scaling (default is False).

    Returns:
    --------
    None
    """
    # Input validation
    if not isinstance(input_vec, np.ndarray):
        raise TypeError("input_vec must be a numpy array.")
    if input_vec.ndim != 1:
        raise ValueError("input_vec must be a 1D array.")
    if saveflag and not output_dir.is_dir():
        raise FileNotFoundError(f"The output directory {output_dir} does not exist.")

    # Determine binning
    unique_values = np.unique(input_vec)
    if len(unique_values) > 20:
        bins = 20
        hist_values, bin_edges = np.histogram(input_vec, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        values, counts = np.unique(input_vec, return_counts=True)
        hist_values = counts
        bin_centers = values

    normalized_counts = hist_values / hist_values.sum()  # Normalize counts

    fig, ax1 = plt.subplots()

    # Handle X-ticks dynamically
    x_ticks = bin_centers
    plt.xticks(
        ticks=x_ticks,
        # labels=[f"{int(v)}" if v < 100 else f"{int(v):,}" for v in x_ticks],
        rotation=45
    )

    # Absolute count bars
    bin_width = 1 if len(bin_centers) <= 1 else 0.8 * (bin_centers[1] - bin_centers[0])
    ax1.bar(bin_centers, hist_values, width=bin_width, color='blue', alpha=0.7, label='Absolute Count')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Absolute Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Normalized counts
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, normalized_counts, color='orange', marker='o', linestyle='-', label='Normalized Portion')
    ax2.set_ylabel('Normalized Portion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    if log_y:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        hist_values = np.maximum(hist_values, 1e-10)
        normalized_counts = np.maximum(normalized_counts, 1e-10)

    # Grid, legend, and title
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9 if title else 1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    print(f'Histogram:\nValues: {bin_centers}\nCounts: {hist_values}\nNormalized Counts: {normalized_counts.round(3)}')

    if saveflag:
        save_path = output_dir / f'Histogram_{title}.png'
        fig.savefig(save_path, dpi=300)
        print(f'Histogram saved to {save_path}')

    if visualize:
        plt.show()


def smart_image_pie_chart(image: np.ndarray, visualize:bool = True) -> None:
    """
    Generate "smart" pie charts from an input image by dynamically handling bins 
    based on the number of unique values and normalizing data for clarity.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array.
        - Single-channel (grayscale) image: 2D array of shape (H, W).
        - RGB image: 3D array of shape (H, W, 3).

    Returns
    -------
    None
        Displays the generated pie chart(s) directly via matplotlib.
    """

    def generate_pie_chart(data: np.ndarray, title: str) -> None:
        """
        Generate a single pie chart for the given 1D pixel data, dynamically determining 
        bins based on the number of unique values.
        
        Parameters
        ----------
        data : np.ndarray
            1D array of values (e.g., flattened pixels).
        title : str
            Title to display on the pie chart.
        """
        unique_values = np.unique(data)
        
        # Determine binning strategy
        if len(unique_values) > 20:
            # Use 20 bins for continuous data
            bins = 20
            hist_values, bin_edges = np.histogram(data, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
            normalized_counts = hist_values / hist_values.sum()  # Normalize counts
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]
        else:
            # Use unique values for discrete data
            values, counts = np.unique(data, return_counts=True)
            hist_values = counts
            bin_centers = values
            normalized_counts = hist_values / hist_values.sum()  # Normalize counts
            bin_labels = [str(value) for value in values]

        # Handle small bins with an "Other" category
        significant_indices = normalized_counts >= 0.01  # Threshold: 1%
        grouped_counts = hist_values[significant_indices]
        grouped_labels = np.array(bin_labels)[significant_indices].tolist()
        
        # Group bins below threshold into "Other"
        if not np.all(significant_indices):  # Only add "Other" if there are insignificant bins
            other_count = np.sum(hist_values[~significant_indices])
            grouped_counts = np.append(grouped_counts, other_count)
            grouped_labels.append("Other")

        # Plot pie chart
        wedges, texts, autotexts = plt.pie(grouped_counts, autopct='%1.1f%%')

        # Add legend with labels
        plt.legend(wedges, grouped_labels, title="Bins", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"Pie Chart for {title}")
        plt.tight_layout()


    # Main logic: Check image dimensions
    if len(image.shape) == 2:
        # Single-channel (grayscale)
        data = image.flatten()
        generate_pie_chart(data, "Single-Channel Image")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # 3-channel RGB
        channel_names = ["Red", "Green", "Blue"]
        for i, ch_name in enumerate(channel_names):
            data = image[..., i].flatten()
            generate_pie_chart(data, f"{ch_name} Channel")
    else:
        raise ValueError(
            "Unsupported image format. Must be single-channel (H, W) or 3-channel (H, W, 3)."
        )
    if visualize:
        plt.show()

def display_unwrapped_single_band_images(subplot_images: tuple[np.ndarray], 
                             titles: tuple[str],
                             key_str: str,
                             output_dir: Path,
                             colormap: str = 'plasma', 
                             saveflag: bool = False,
                             visualize: bool = True,
                             canvas_size: list = [540, 1440],
                             v_fov: list = [0, 135],
                             h_fov: list = [0, 360]
                             ) -> None:
    """
    Display images with titles and colorbars.

    Parameters:
    subplot_images (np.ndarray): A tuple of images to display.
    titles (str): Titles for the images.
    colormap (str): Colormap to use for Generating the images. Default is 'plasma'.
    saveflag (bool): If True, save the images to the output_dir. Default is False.

    Returns:
    None
    """
    CANVAS_HEIGHT, CANVAS_WIDTH = canvas_size
    num_subplots = len(subplot_images)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4*num_subplots))

    # Set the colormap to something more visually friendly
    colormap = colormap  # e.g.: 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # Define custom ticks using np.linspace
    y_ticks = np.linspace(v_fov[0], v_fov[1], CANVAS_HEIGHT + 1)
    x_ticks = np.linspace(h_fov[0], h_fov[1], CANVAS_WIDTH + 1)
    for ax, subplot_img, title in zip(axes, subplot_images, titles):
        print(f"Generating {title} image...")
        im = ax.imshow(subplot_img, cmap=colormap, aspect='auto', extent=[x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]])
        ax.set_xlabel('Azimuth Angle (degrees)')
        ax.set_ylabel('Elevation Angle (degree)')
        ax.set_xticks(x_ticks[::int(len(x_ticks) / 10)])  # Reduce the number of x-ticks to avoid overlap
        ax.set_yticks(y_ticks[::int(len(y_ticks) / 10)])  # Reduce the number of y-ticks for readability
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Adjust colorbar size
        ax.set_title(title)
    
    plt.tight_layout()

    if saveflag:
        for i, title in enumerate(titles):
            plt.imsave(output_dir / f'{title}_{key_str}.png', subplot_images[i], cmap=colormap)
        
        fig.savefig(output_dir / f'combined_unwrapped_images_{key_str}.png')
        print(f'Images saved to {output_dir} directory')

    if visualize:
        plt.show()




def display_single_band_img_wt_discrete_values(
    image_data: np.ndarray,
    output_dir: Path,
    title: str = "Point Density Map",
    simpl_colormap: str = 'jet',
    saveflag: bool = False,
    visualize: bool = True,
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

    # Clip values to match color bins (0 to 9)
    unique_values = np.clip(unique_values, 0, 9).astype(int)

    # Define color boundaries and colormap
    boundaries = list(range(10)) + [1e6]  # Bins: [0–1), [1–2), ..., [9–inf)
    jet = plt.cm.get_cmap('jet', len(boundaries) - 1)
    colors = [jet(i) for i in range(jet.N)] + ['gray']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    # --- Display the image ---
    im = ax.imshow(image_data, cmap=cmap, norm=norm)

    # --- Compute frequencies and cumulative proportions for proportional colorbar ---
    flat_data = image_data.flatten()
    flat_data_clipped = np.clip(flat_data, 0, 9).astype(int)
    value_counts = np.array([np.count_nonzero(flat_data_clipped == i) for i in range(10)])
    proportions = value_counts / value_counts.sum()
    cumulative = np.concatenate([[0], np.cumsum(proportions)])

    # Create separate axes for custom colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    # --- Build proportional colorbar ---
    cb = ColorbarBase(
        cax,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(cumulative, cmap.N),
        ticks=(cumulative[:-1] + cumulative[1:]) / 2,
        spacing='proportional',
        orientation='vertical'
    )

    # Set colorbar tick labels
    cb.ax.set_yticklabels([str(i) for i in range(10)])
    cb.set_label('Points per pixel')
 
    if saveflag:
        # Save the raw image without labels or colorbars
        normalized_data = norm(image_data)
        rgba_image = cmap(normalized_data)
        plt.imsave(output_dir / f'{title}.png', rgba_image, format='png', dpi=1)

        # # Save the displayed image with annotations
        # fig.savefig(output_dir / f'{title}.png', dpi=600)
        print(f'Images saved to {output_dir} directory')

    get_image_histogram(image_data=image_data, 
                        title=title, 
                        saveflag=saveflag, 
                        output_dir=output_dir,
                        visualize=visualize
                        )
    if visualize:
        smart_image_pie_chart(image_data)
        plt.show()


def display_unwrapped_rgb_image(rgb_image: np.ndarray, 
                            figure_title: str, 
                            output_dir: Path, 
                            saveflag: bool = False,
                            visualize: bool = True,
                            canvas_size: list = [540, 1440],
                            v_fov: list = [0, 135],
                            h_fov: list = [0, 360]
                            ) -> None:
    """
    Display the unwrapped RGB image with custom ticks.
    """
    CANVAS_HEIGHT, CANVAS_WIDTH = canvas_size
    y_ticks = np.linspace(v_fov[0], v_fov[1], CANVAS_HEIGHT + 1)
    x_ticks = np.linspace(h_fov[0], h_fov[1], CANVAS_WIDTH + 1)
    y_label = 'Elevation Angle (degree)'
    plt.figure(figsize=(12, 6))
    plt.imshow(rgb_image, aspect='auto', extent=[x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]])
    
    plt.title(figure_title)
    plt.xticks(x_ticks[::int(len(x_ticks) / 10)])  # Reduce the number of x-ticks to avoid overlap
    plt.yticks(y_ticks[::int(len(y_ticks) / 10)])  # Reduce the number of y-ticks for readability
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel(y_label)

    if saveflag:
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        io.imsave(f'{output_dir}/{figure_title}.png', rgb_image_uint8)

    if visualize:
        smart_image_pie_chart(rgb_image)
        plt.show()


def plot_correlation_matrix(corr_matrix, 
                            band_names=None, 
                            title="Correlation Matrix of Feature Maps",
                            output_dir:Path=None, 
                            output_stem=None):
    """
    Plots the correlation matrix as a heatmap.

    Parameters:
    -----------
    corr_matrix : np.ndarray
        A (C, C) correlation matrix.

    band_names : list of str, optional
        A list of names for the spectral bands (length C). If None, band indices will be used.

    title : str
        Title of the heatmap.
    """
    C = corr_matrix.shape[0]
    if band_names is None:
        band_names = [f'Band {i}' for i in range(C)]

    assert len(band_names) == C, f"Length of band names {len(band_names)} must match the number of bands {C}."
    corr_fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=band_names, yticklabels=band_names,
                square=True, cbar_kws={"shrink": 0.75})
    plt.title(title)
    plt.tight_layout()

    if output_dir is None:
        output_path = Path(f"outputs/correlation_matrix_{output_stem}.png")
    else:
        output_path = output_dir / f"correlation_matrix_{output_stem}.png"
    corr_fig.savefig(output_path)
    print(f"1️Correlation matrix saved to {output_path}")


def plot_pca_components(pcs, output_dir:Path=None, output_stem:str = None):
    """
    Plots the PCA/ICA/MNF components as single channel images.

    Parameters:
    -----------
    pcs : np.ndarray
        Principal components of shape (n_components, H, W).
    """
    n_components = pcs.shape[-1]
    fig, axes = plt.subplots(n_components, 1, figsize=(6, 3 * n_components))
    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        axes[i].imshow(pcs[:,:, i], cmap='plasma') # cmaps: 'gray', 'hot', 'cool', 'viridis', 'plasma', 'inferno'
        axes[i].set_title(f'Component {i+1}')
        axes[i].axis('off')

    if 'PCA' in output_stem:
        plt.suptitle("PCA Components")
    elif 'MNF' in output_stem:
        plt.suptitle("MNF Components")
    elif 'ICA' in output_stem:
        plt.suptitle("ICA Components")
    else:
        plt.suptitle("Unknown Components")
    plt.tight_layout()

    if output_dir is None:
        output_path = Path(f"outputs/pca_components_{output_stem}.png")
    else:
        output_path = output_dir / f"pca_components_{output_stem}.png"
    fig.savefig(output_path)
    print(f"2️PCA/MNF/ICA components saved to {output_path}")
    

def plot_rgb_permutations(components, output_dir:Path=None, output_stem:str=None):
    """
    Plots RGB images from all permutations of the first 3 components.

    Parameters:
    -----------
    components : np.ndarray
        Component images of shape (3, H, W).
    title : str
        Title prefix for each subplot.
    """
    from itertools import permutations

    permuts = list(permutations([0, 1, 2]))
    H, W = components.shape[:2]
    fig, axes = plt.subplots(len(permuts), 1, figsize=(6, 3 * len(permuts)))

    for ax, perm in zip(axes, permuts):
        rgb = np.stack([components[:,:,i] for i in perm], axis=-1)
        # Normalize each channel
        for i in range(3):
            ch = rgb[:, :, i]
            rgb[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

        # save rgb image
        rgb = (rgb * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb)
        if output_dir is None:
            output_path = Path(f"outputs/{output_stem}_rgb_{perm[0]}_{perm[1]}_{perm[2]}.png")
        else:
            output_path = output_dir / f"{output_stem}_rgb_{perm[0]}_{perm[1]}_{perm[2]}.png"   
        rgb_img.save(output_path)
        print(f"3️Saved RGB image to {output_path}")
        ax.imshow(rgb)
        ax.set_title(f"{output_stem} rgb permutations\nR:Comp{perm[0]+1} G:Comp{perm[1]+1} B:Comp{perm[2]+1}")
        ax.axis('off')

    plt.tight_layout()
    
    if output_stem:
        if output_dir is None:
            output_path = Path(f"outputs/{output_stem}_PCs_permutations.png")
        else:
            output_path = output_dir / f"{output_stem}_PCs_permutations.png"
        fig.savefig(output_path)
        print(f"3️Saved RGB permutations plot to {output_path}")


def histogram_to_ascii(hist, width=30, style="blocks"):
    if style == "blocks":
        # Unicode blocks for smooth gradients
        bars = "▁▂▃▄▅▆▇█"
    else:
        # ASCII fallback
        bars = " .:-=+*#%@"

    max_count = max(hist)
    if max_count == 0:
        return "".join([" " for _ in hist])

    result = ""
    for count in hist:
        bar_index = int((count / max_count) * (len(bars) - 1))
        result += bars[bar_index]
    return result