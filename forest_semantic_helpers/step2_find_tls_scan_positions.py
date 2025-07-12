import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from skimage.feature import blob_log
from pathlib import Path

def compute_density_map(points, bins=512):
    """Compute 2D density map from point cloud."""
    x, y = points[:, 0], points[:, 1]
    density, xedges, yedges, _ = binned_statistic_2d(
        x, y, None, statistic='count', bins=bins
    )
    return density.T, xedges, yedges


def preprocess_density(density):
    """Normalize and invert density map."""
    density = np.nan_to_num(density)
    norm = (density.max() - density)
    return ((norm - norm.min()) / (norm.ptp() + 1e-6) * 255).astype(np.uint8)


def detect_tls_positions(density_img, min_sigma=10, max_sigma=15, threshold=0.1):
    """Detect TLS positions using blob detection."""
    return blob_log(density_img, min_sigma=min_sigma, max_sigma=max_sigma,
                   num_sigma=10, threshold=threshold)


def blobs_to_coords(blobs, xedges, yedges):
    """Convert blob pixel coordinates to real coordinates."""
    coords = []
    for ypix, xpix, _ in blobs:
        x = (xedges[int(xpix)] + xedges[int(xpix) + 1]) / 2
        y = (yedges[int(ypix)] + yedges[int(ypix) + 1]) / 2
        coords.append((x, y))
    return np.array(coords)


def visualize_results(density_img, blobs, output_dir):
    """Create visualization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Density map
    im1 = ax1.imshow(density_img, cmap='viridis', origin='lower')
    ax1.set_title('Density Map')
    plt.colorbar(im1, ax=ax1, label='Density (inverted)')
    
    # Density map with blobs
    im2 = ax2.imshow(density_img, cmap='viridis', origin='lower')
    ax2.set_title(f'Detected Blobs ({len(blobs)})')
    for ypix, xpix, radius in blobs:
        circle = plt.Circle((xpix, ypix), radius, color='red', fill=False, linewidth=2)
        ax2.add_patch(circle)
        ax2.plot(xpix, ypix, 'r.', markersize=8)
    plt.colorbar(im2, ax=ax2, label='Density (inverted)')
    fig.savefig(output_dir / "density_map_with_blobs.png", dpi=300)


    

def main():
    output_dir = Path("/home/fzhcis/Downloads/ForestSemantic/output")
    pts_path = output_dir / "Plot_1_ground_sample_points.csv"

    if not pts_path.exists():
        raise FileNotFoundError(f"File not found: {pts_path}")
    
    points_df = pd.read_csv(pts_path)
    points = points_df[['X', 'Y', 'Z']].values

    # Process
    density, xedges, yedges = compute_density_map(points[:, :2])
    density_img = preprocess_density(density)
    blobs = detect_tls_positions(density_img)
    tls_positions = blobs_to_coords(blobs, xedges, yedges)
    
    print(f"Detected {len(tls_positions)} TLS positions")
    
    # Visualize
    visualize_results(density_img, blobs, output_dir)

    # Save results
    output_path = output_dir/ "tls_positions_xy.csv"
    pd.DataFrame(tls_positions, columns=['X', 'Y']).to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
            


if __name__ == "__main__":
    main()

