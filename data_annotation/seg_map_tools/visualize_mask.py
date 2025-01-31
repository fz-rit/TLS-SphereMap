import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

def visualize_mask(image_path, cmap='gray', save=False):
    # Load the grayscale segmentation mask
    image_path = Path(image_path)
    mask = np.array(Image.open(image_path))
    print("---Checkout the label_maps.json file to see the class index for each class.---")
    # Create a figure, assign figure size according to the shape of the mask 
    fig_size = (mask.shape[1] / 100, mask.shape[0] / 100)
    plt.figure(figsize=fig_size)
    
    # Display with a colormap
    plt.imshow(mask, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Class Index")

    # Remove axes
    plt.axis("off")

    # Save the visualization if the user specifies -s or --save
    if save:
        save_path = os.path.join(image_path.parent, f"{image_path.stem}_visualized.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Visualization saved as {save_path}")

    # Show the image
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a segmentation mask with a colormap.")
    parser.add_argument("-i", "--input", required=True, help="Path to the segmentation mask image (grayscale PNG).")
    parser.add_argument("-c", "--cmap", default="gray", help="Colormap to use for visualization.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the visualization as visualized_mask.png")

    args = parser.parse_args()
    visualize_mask(image_path = args.input, 
                    cmap = args.cmap,
                    save = args.save)
