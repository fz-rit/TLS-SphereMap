import numpy as np
from illustrate_spherical_projection import (generate_lidar_ball, 
                                             visualize_point_cloud,
                                             spherical_projection_with_density,
                                             visualize_and_save_point_cloud, 
                                             create_colored_cube_points)
from PIL import Image
import pandas as pd

def image_preprocess(image, img_size):
    """
    Preprocess the image for spherical projection.
    Resizes the image to img_size and converts it to RGB format.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure shape (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # grayscale to RGB

    if image.shape[0] == 3:  # (H, W, 3) -> (3, H, W) 
        image = np.transpose(image, (1, 2, 0))

    # Resize to (271, 720) if needed (zenith: 0–135 at 0.5° → 271 rows)
    if image.shape[0] != img_size[0] or image.shape[1] != img_size[1]:
        image = np.array(Image.fromarray(image).resize((img_size[1], img_size[0]), resample=Image.BILINEAR))

    # Convert to float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)


    print("Image Shape:", image.shape)
    print("Image Dtype:", image.dtype)
    print("Image Max:", image.max())
    return image



def back_project_color_to_ball(df_ball, image, zenith_range= (0, 135), inverse_zenith=False):
    """
    Assigns image color to df_ball using azimuth and zenith angles.
    Assumes image shape (271, 720), angular res = 0.5°, azimuth 0–360, zenith 0–135.
    """
    img_size = int((zenith_range[1] - zenith_range[0]) // 0.5 + 1), 720  # (271, 720) for zenith range (0, 135)
    image = image_preprocess(image, img_size)

    # Compute row, col indices
    zenith_deg_shifted = df_ball['zenith_deg'] - df_ball['zenith_deg'].min()
    row = (zenith_deg_shifted / 0.5).astype(int)
    col = (df_ball['azimuth_deg'] / 0.5).astype(int)
    # Adjust for inverse zenith
    if inverse_zenith:
        row = img_size[0] - 1 - row

    # Get RGB values from image (normalize if uint8)
    rgb = image[row, col] / 255.0 if image.dtype == np.uint8 else image[row, col]

    # Assign to DataFrame
    df_ball = df_ball.copy()
    df_ball['r'] = rgb[:, 0]
    df_ball['g'] = rgb[:, 1]
    df_ball['b'] = rgb[:, 2]

    return df_ball


# ZENITH_RANGE = (75, 105)  # Zenith range for the ball
# OUTPUT_PREFIX = "VLP64"

ZENITH_RANGE = (0, 135)  # Zenith range for the ball
# OUTPUT_PREFIX = "palau_2024"
OUTPUT_PREFIX = "harvard_forest"

df_ball = generate_lidar_ball(radius=10.0, zenith_range=ZENITH_RANGE)
# visualize_point_cloud(df_ball, add_cube=True)
# visualize_and_save_point_cloud(df_ball, zenith_range=ZENITH_RANGE, visualize=True, output_prefix=OUTPUT_PREFIX+"_ball")
# image_path = '/home/fzhcis/mylab/data/fall_at_harvard_forest.jpg'
# image_path = '/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/pca_outputs/6962/UMBCBL009_868796962_image_cube_ICA_rgb_0_1_2.png'
# image_path = '/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/pca_outputs/6962/UMBCBL009_868796962_image_cube_PCA_rgb_1_2_0.png'
# image_path = '/home/fzhcis/mylab/data/street_panorama_south_north.png'
# image_path = '/home/fzhcis/mylab/data/AdobeStock_156245359.jpeg'
image_path = '/home/fzhcis/mylab/data/harvard_forest_snapshot.png'
rgb_img = np.array(Image.open(image_path))  # shape (H, W, 3)

# rgb_img, density_map = spherical_projection_with_density(df_ball, zenith_range=ZENITH_RANGE)

print("RGB Image Shape:", rgb_img.shape)
print("RGB Image Dtype:", rgb_img.dtype)
print("RGB Image Max:", rgb_img.max())


df_colored = back_project_color_to_ball(df_ball, rgb_img, 
                                        zenith_range=ZENITH_RANGE,
                                        inverse_zenith=False)

df_cube = create_colored_cube_points(center=[0, 0, 0], size=0.5, samples_per_face=50)
df_combined = pd.concat([df_colored, df_cube], ignore_index=True)
visualize_and_save_point_cloud(df_combined, zenith_range=ZENITH_RANGE, visualize=True, output_prefix=OUTPUT_PREFIX+"_with_cube")
