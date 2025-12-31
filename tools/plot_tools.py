
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from skimage import io
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from PIL import Image
import pandas as pd
import plotly.express as px

plt.rcParams.update({
    'font.size': 12,         # base font size
    'axes.titlesize': 12,    # title size
    'axes.labelsize': 10,    # x/y label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})
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
                        visualize: bool = True,
                        cmap: ListedColormap = None
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
    
    # Smart binning: if more than 10 unique values, keep top 9 and group rest as "Other"
    if len(values) > 10:
        # Sort by counts in descending order
        sorted_indices = np.argsort(counts)[::-1]
        top_9_indices = sorted_indices[:9]
        other_indices = sorted_indices[9:]
        
        # Keep top 9 values and their counts
        top_values = values[top_9_indices]
        top_counts = counts[top_9_indices]
        
        # Sum counts for "Other" category
        other_count = np.sum(counts[other_indices])
        
        # Combine top values with "Other"
        final_values = np.append(top_values, "Other")
        final_counts = np.append(top_counts, other_count)
        
        # Create labels for x-axis
        value_labels = [f"{int(v)}" if v < 100 else f"{int(v):,}" for v in top_values] + ["Other"]
    else:
        final_values = values
        final_counts = counts
        value_labels = [f"{int(v)}" if v < 100 else f"{int(v):,}" for v in values]
    
    normalized_counts = final_counts / final_counts.sum()  # Normalize counts

    # Map values to evenly spaced indices for plotting
    x_indices = np.arange(len(final_values))

    # Create the bar plot
    fig, ax1 = plt.subplots(figsize=(4.5, 2.5), dpi=300)

    if cmap:
        colors = [cmap(i / (len(x_indices)-1)) for i in x_indices]
    else:
        colors = 'blue'

    ax1.bar(x_indices, final_counts, color=colors, alpha=0.7, label='Absolute Count')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Absolute Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # alternative syntax
    # add values on top of bars
    for x, y in zip(x_indices, final_counts):
        ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=8)

    # Add a second y-axis for normalized portion
    ax2 = ax1.twinx()
    ax2.plot(x_indices, normalized_counts, color='orange', marker='o', linestyle='-', label='Normalized Portion')
    ax2.set_ylabel('Normalized Portion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Replace x-ticks with the actual pixel values
    plt.xticks(x_indices, labels=value_labels, rotation=45)

    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.title(f'Histogram of {title}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    
    # Save the figure if saveflag is True
    if saveflag:
        save_path = output_dir / f'Histogram_{title}.pdf'
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Histogram saved to {save_path}')
    
    if visualize:
        plt.show()
    else:
        plt.close('all')


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
    for x, y in zip(bin_centers, hist_values):
        ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=8)
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

    if saveflag:
        save_path = output_dir / f'Histogram_{title}.png'
        fig.savefig(save_path, dpi=300)
        print(f'Histogram saved to {save_path}')

    if visualize:
        plt.show()
    else:
        plt.close('all')

def generate_pie_chart(counts,
    labels,
    title,
    explode_label=None,
    save_path=None,
):
    df = pd.DataFrame({"label": labels, "count": counts})
    df["pull"] = 0.0

    if explode_label is not None:
        df.loc[df["label"] == explode_label, "pull"] = 0.12

    fig = px.pie(
        df,
        values="count",
        names="label",
        hole=0.4,  # donut = pseudo 3D
        color="label",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        pull=df["pull"],
        marker=dict(line=dict(color="black", width=1)),
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    if save_path:
        fig.write_image(save_path, width=800, height=600)
    else:
        fig.show()




# def smart_image_pie_chart(grouped_counts, grouped_labels,
#                           title: str,
#                           visualize: bool = True,
#                           save_path: Path = None) -> None:
#     """
#     Generate pie charts for image pixel value distributions.
#     """
    
#     generate_pie_chart
#     if visualize:
#         plt.show()
#     else:
#         plt.close('all')

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
    else:
        plt.close('all')


def _determine_colormap_range(image_data: np.ndarray, color_map: dict = None) -> tuple[int, list]:
    """
    Determine the appropriate range and colors for discrete value visualization.
    
    Parameters
    ----------
    image_data : np.ndarray
        The image data containing discrete integer values.
    color_map : dict, optional
        Dictionary mapping string keys to color values.
        
    Returns
    -------
    tuple[int, list]
        Number of unique values and list of colors to use.
    """
    image_unique_values, pixel_val_counts = np.unique(image_data, return_counts=True)
    image_unique_values = image_unique_values.astype(int)
    
    if color_map is not None and isinstance(color_map, dict):
        # Get available color map keys as integers
        colormap_keys = set()
        for key in color_map.keys():
            try:
                colormap_keys.add(int(key))
            except (ValueError, TypeError):
                continue
        
        print(f"Available color map keys: {sorted(colormap_keys)}")
        
        # Check if image values are within color map range
        image_values_in_colormap = set(image_unique_values) & colormap_keys
        
        if image_values_in_colormap:
            num_unique_values = max(colormap_keys) + 1
            print(f"Using color map range: 0 to {num_unique_values-1}")
        else:
            num_unique_values = max(image_unique_values) + 1
            print(f"Image values don't match color map, using image range: 0 to {num_unique_values-1}")
    else:
        num_unique_values = max(image_unique_values) + 1
        print(f"No color map provided, using image range: 0 to {num_unique_values-1}")
    
    # Ensure we have at least the minimum needed for image data
    num_unique_values = max(num_unique_values, max(image_unique_values) + 1)
    
    # Generate colors
    if color_map is None:
        tab10 = plt.cm.get_cmap('tab10', num_unique_values)
        colors = ['gray'] + [tab10(i) for i in range(num_unique_values)]
    elif isinstance(color_map, dict):
        colors = ['gray']
        default_cmap = plt.cm.get_cmap('tab10', num_unique_values)
        for i in range(num_unique_values):
            if str(i) in color_map:
                colors.append(color_map[str(i)])
            else:
                colors.append(default_cmap(i))
    else:
        raise ValueError("color_map must be None or a dictionary")
    
    return num_unique_values, colors, pixel_val_counts


# def _create_proportional_colorbar(fig, ax, colors: list, num_unique_values: int, 
#                                  image_data: np.ndarray, cb_label: str) -> None:
#     """
#     Create a proportional colorbar based on value frequencies in the image.
    
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         The figure to add the colorbar to.
#     ax : matplotlib.axes.Axes
#         The main axes containing the image.
#     colors : list
#         List of colors for the colorbar.
#     num_unique_values : int
#         Number of unique values in the data.
#     image_data : np.ndarray
#         The image data for computing frequencies.
#     cb_label : str
#         Label for the colorbar.
#     """
#     # Compute frequencies and cumulative proportions
#     flat_data = image_data.flatten()
#     flat_data_clipped = np.clip(flat_data, 0, num_unique_values-1).astype(int)
#     value_counts = np.array([np.count_nonzero(flat_data_clipped == i) for i in range(num_unique_values)])
#     proportions = value_counts / value_counts.sum()
#     cumulative = np.concatenate([[0], np.cumsum(proportions)])

#     # Create separate axes for custom colorbar
#     cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

#     # Build proportional colorbar
#     cb = ColorbarBase(
#         cax,
#         cmap=ListedColormap(colors[:-1]),  # Exclude 'gray' for overflow values
#         norm=BoundaryNorm(cumulative, len(colors)-1),
#         ticks=(cumulative[:-1] + cumulative[1:]) / 2,
#         spacing='proportional',
#         orientation='vertical'
#     )

#     cb.ax.set_yticklabels([str(i) for i in range(num_unique_values)])
#     cb.set_label(cb_label)

    


# def display_single_band_img_wt_discrete_values(
#     image_data: np.ndarray,
#     output_dir: Path,
#     color_map: dict = None,
#     title: str = "Point Density Map",
#     cb_label: str = "Point Density",
#     saveflag: bool = False,
#     visualize: bool = True,
# ) -> None:
#     """
#     Display a single-band image with discrete values using automatic colormap determination.

#     Parameters
#     ----------
#     image_data : np.ndarray
#         The image data to display, expected to contain discrete integer values.
#     output_dir : Path
#         The directory where the image will be saved if `saveflag` is True.
#     color_map : dict, optional
#         Dictionary mapping string keys to color values. If None, uses default colormap.
#     title : str, optional
#         The title of the image, by default "Point Density Map".
#     cb_label : str, optional
#         Label for the colorbar, by default "Point Density".
#     saveflag : bool, optional
#         If True, saves the image to the output directory, by default False.
#     visualize : bool, optional
#         If True, displays the image, by default True.
#     """
#     # Determine colormap range and colors
#     num_unique_values, colors, pixel_val_counts = _determine_colormap_range(image_data, color_map)
    
#     # Setup figure and display
#     fig, ax = plt.subplots(figsize=(18, 5))
#     boundaries = list(range(num_unique_values)) + [1e6]
#     custom_colormap = ListedColormap(colors)
#     bnd_norm = BoundaryNorm(boundaries, ncolors=custom_colormap.N)
    
#     # Display the image
#     im = ax.imshow(image_data, cmap=custom_colormap, norm=bnd_norm)
    
#     # Save image if requested
#     if saveflag:
#         normalized_data = bnd_norm(image_data)
#         rgba_image = custom_colormap(normalized_data)
#         plt.imsave(output_dir / f'{title}.png', rgba_image, format='png', dpi=1)
#         print(f'Images saved to {output_dir} directory')

#     get_image_histogram(
#         image_data=image_data, 
#         title=title, 
#         saveflag=saveflag, 
#         output_dir=output_dir,
#         visualize=visualize,
#         cmap=custom_colormap,
#     )
    
#     if visualize:
#         smart_image_pie_chart(image_data)
#         plt.show()
#     else:
#         plt.close('all')


def save_fig_with_legend(
    image_data,
    output_path,
    *,
    title=None,
    cmap=None,
    norm=None,
    class_labels=None,
    class_values=None,
    class_colors=None,
    legend_loc="lower right",
    figsize=(10, 4),
    dpi=300,
    show_axis=False,
):
    """
    Save an imshow figure with a discrete legend.

    Parameters
    ----------
    image_data : ndarray (H, W)
        Image or label map to visualize.
    output_path : str or Path
        Output PNG path.
    title : str, optional
        Figure title.
    cmap : matplotlib colormap
        Colormap used by imshow.
    norm : matplotlib Normalize
        Normalization used by imshow.
    class_labels : list[str]
        Labels shown in legend (e.g. ['0', '1', '2', '>2']).
    class_values : list
        Representative values for color lookup (e.g. [0, 1, 2]).
        Used only if class_colors is None.
    class_colors : list
        Explicit colors for legend (overrides class_values).
    legend_loc : str
        Legend location.
    figsize : tuple
        Figure size.
    dpi : int
        Output DPI.
    show_axis : bool
        Whether to show axes.
    """
    import matplotlib.patches as mpatches
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        image_data,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    if title:
        ax.set_title(title)

    if not show_axis:
        ax.axis("off")

    # ---------------- Legend construction ----------------
    if class_labels is not None:

        if class_colors is None:
            assert cmap is not None and norm is not None and class_values is not None, (
                "If class_colors is not provided, cmap, norm, and class_values are required."
            )

            class_colors = [
                cmap(norm(v)) for v in class_values
            ]

        legend_patches = [
            mpatches.Patch(color=c, label=l)
            for c, l in zip(class_colors, class_labels)
        ]

        ax.legend(
            handles=legend_patches,
            loc=legend_loc,
            frameon=True,
            framealpha=1.0,
            fontsize=14,
        )

    # ---------------- Save ----------------
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close(fig)



def display_single_band_img_wt_discrete_values(
    image_data: np.ndarray,
    output_dir: Path,
    color_map: dict = None,
    title: str = "pt_density_map",
    saveflag: bool = False,
    visualize: bool = True,
) -> None:
    """
    Display a single-band image with discrete values using automatic colormap determination.

    Parameters
    ----------
    image_data : np.ndarray
        The image data to display, expected to contain discrete integer values.
    output_dir : Path
        The directory where the image will be saved if `saveflag` is True.
    color_map : dict, optional
        Dictionary mapping string keys to color values. If None, uses default colormap.
    title : str, optional
        The title of the image, by default "pt_density_map".
    saveflag : bool, optional
        If True, saves the image to the output directory, by default False.
    visualize : bool, optional
        If True, displays the image, by default True.
    """
    # Determine colormap range and colors
    num_unique_values, colors, pixel_val_counts = _determine_colormap_range(image_data, color_map)
    
    if "density" in title.lower():
        # merge values > 2 into "Other": 0-no points, 1-single return, 2-dual return, >2-multiple points
        if num_unique_values > 4:
            num_unique_values = 4
            colors = colors[:4]  # Keep only first 4 colors: 0, 1, 2, >2
            pixel_val_counts = pixel_val_counts[:3].tolist() + [np.sum(pixel_val_counts[3:])]
            print(f"Merging values > 2 into 'Other' category.")
            print(f"Updated unique values counts: {dict(zip(range(num_unique_values), pixel_val_counts))}")
            print(f"Updated unique values in portion in percent: {dict(zip(range(num_unique_values), (np.array(pixel_val_counts) / np.sum(pixel_val_counts) * 100).round(1)))}%")

    boundaries = list(range(num_unique_values)) + [1e6]
    custom_colormap = ListedColormap(colors)
    bnd_norm = BoundaryNorm(boundaries, ncolors=custom_colormap.N)
    
    # # Display the image
    # fig, ax = plt.subplots(figsize=(18, 5))
    # im = ax.imshow(image_data, cmap=custom_colormap, norm=bnd_norm)
    
    # Save image if requested
    if saveflag:
        if "density" in title.lower():
            save_fig_with_legend(
                            image_data=image_data,
                            output_path=output_dir / f'{title}.png',
                            # title=title,
                            cmap=custom_colormap,
                            norm=bnd_norm,
                            # class_labels=[str(i) for i in range(num_unique_values)],
                            class_labels = ['0', '1', '2', '>2'],
                            class_values=list(range(num_unique_values)),
                            legend_loc="lower right",
                            figsize=(18, 5),
                            dpi=300,
                            show_axis=False,
                        )
        else:
            normalized_data = bnd_norm(image_data)
            rgba_image = custom_colormap(normalized_data)
            plt.imsave(output_dir / f'{title}.png', rgba_image, format='png', dpi=1)
            print(f'Images saved to {output_dir} directory')


    

    get_image_histogram(
        image_data=image_data, 
        title=title, 
        saveflag=saveflag, 
        output_dir=output_dir,
        visualize=visualize,
        cmap=custom_colormap,
    )
    
    generate_pie_chart(pixel_val_counts, labels=['0', '1', '2', '>2'], title=title, save_path=output_dir / f'PieChart_{title}.png')
    # if visualize:
    #     generate_pie_chart(pixel_val_counts, labels=['0', '1', '2', '>2'], title=title)
    #     plt.show()
    # else:
    #     plt.close('all')


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
        if rgb_image.dtype == np.uint8:
            rgb_image_uint8 = rgb_image
        elif rgb_image.dtype == np.float32 and rgb_image.max() <= 1.0:
            rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        else:
            raise ValueError(f"Must be uint8 or float32 with values in [0, 1]. \
                             Unsupported image dtype {rgb_image.dtype}. \
                             Image values: min={rgb_image.min()}, max={rgb_image.max()}")
        io.imsave(f'{output_dir}/{figure_title}.png', rgb_image_uint8)

    if visualize:
        values, counts = np.unique(rgb_image, return_counts=True)
        hist_values = counts
        bin_labels = [str(value) for value in values]
        generate_pie_chart(counts=hist_values, labels=bin_labels, title=figure_title)
        plt.show()
    else:
        plt.close('all')



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
    plt.close('all')

def plot_components_permutations(components, output_dir:Path=None, output_stem:str=None):
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
        
        if perm == (1, 2, 0):
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

    plt.close('all')

        
def plot_correlation_matrix(corr_matrix, 
                            band_names=None, 
                            output_dir: Path = None, 
                            output_stem: str = None,
                            dpi: int = 300):
    """
    Plots a hybrid-style correlation matrix using Seaborn heatmap + bubble overlay.

    Parameters
    ----------
    corr_matrix : np.ndarray or pd.DataFrame
        Correlation matrix of shape (C, C).
    band_names : list of str, optional
        Labels for axes. If None, band indices are used.
    output_dir : Path, optional
        Directory to save the figure.
    output_stem : str, optional
        File name stem for saving.
    dpi : int
        Resolution for saved figure.
    """
    corr = corr_matrix if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame(corr_matrix)
    C = corr.shape[0]
    
    if band_names is None:
        band_names = [f'Band {i}' for i in range(C)]

    assert len(band_names) == C, f"Length of band names {len(band_names)} must match number of bands {C}."

    # ─────────────────────────────────────────────
    # Setup
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=dpi)
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.colormaps["coolwarm"]

    # ─────────────────────────────────────────────
    # Lower triangle: heatmap with annotations
    sns.heatmap(corr,
                mask=mask_upper,
                cmap=cmap,
                vmin=-1, vmax=1,
                annot=True, fmt=".2f",
                square=True,
                linewidths=0.5,
                xticklabels=band_names,
                yticklabels=band_names,
                cbar_kws={"shrink": 0.75, "label": "Correlation coefficient"},
                annot_kws={"size": 7},
                ax=ax)

    # ─────────────────────────────────────────────
    # Upper triangle: bubble glyph overlay
    max_bubble_area = 800
    for i in range(C):
        for j in range(i+1, C):
            val = corr.iloc[i, j]
            radius = abs(val)  # perceptual scaling
            area = max_bubble_area * radius ** 2
            ax.scatter(j + 0.5, i + 0.5,
                       s=area,
                       color=cmap(norm(val)),
                       edgecolor='white',
                       linewidth=0.5,
                       alpha=0.8)

    # ─────────────────────────────────────────────
    # Aesthetics
    ax.set_xticklabels(band_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(band_names, rotation=0, fontsize=9)
    ax.tick_params(length=0)
    plt.tight_layout()

    # ─────────────────────────────────────────────
    # Save to file
    if output_stem is None:
        output_stem = "corr"
    if output_dir is None:
        output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"correlation_matrix_{output_stem}.png"
    fig.savefig(output_path)
    print(f"✅ Correlation matrix saved to: {output_path}")
    plt.close('all')


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