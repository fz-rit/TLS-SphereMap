import numpy as np
from illustrate_spherical_projection import generate_full_lidar_point_cloud, visualize_point_cloud, spherical_projection_with_density
from PIL import Image
import pandas as pd

def image_preprocess(image):
    """
    Preprocess the image for spherical projection.
    Resizes the image to (271, 720) and converts it to RGB format.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure shape (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # grayscale to RGB

    if image.shape[0] == 3:  # (H, W, 3) -> (3, H, W) 
        image = np.transpose(image, (1, 2, 0))

    # Resize to (271, 720) if needed (zenith: 0–135 at 0.5° → 271 rows)
    if image.shape[0] != 271 or image.shape[1] != 720:
        image = np.array(Image.fromarray(image).resize((720, 271), resample=Image.BILINEAR))

    # Convert to float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)


    # print("Image Shape:", image.shape)
    # print("Image Dtype:", image.dtype)
    # print("Image Max:", image.max())
    return image



def back_project_color_to_ball(df_ball, image, inverse_zenith=False):
    """
    Assigns image color to df_ball using azimuth and zenith angles.
    Assumes image shape (271, 720), angular res = 0.5°, azimuth 0–360, zenith 0–135.
    """
    image = image_preprocess(image)

    # Compute row, col indices
    row = (df_ball['zenith_deg'] / 0.5).astype(int)
    col = (df_ball['azimuth_deg'] / 0.5).astype(int)

    # Adjust for inverse zenith
    if inverse_zenith:
        row = 270 - row

    # Get RGB values from image (normalize if uint8)
    rgb = image[row, col] / 255.0 if image.dtype == np.uint8 else image[row, col]

    # Assign to DataFrame
    df_ball = df_ball.copy()
    df_ball['r'] = rgb[:, 0]
    df_ball['g'] = rgb[:, 1]
    df_ball['b'] = rgb[:, 2]

    return df_ball



df_ball = generate_full_lidar_point_cloud(radius=10.0)

# image_path = '/home/fzhcis/mylab/data/fall_at_harvard_forest.jpg'
image_path = '/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/pca_outputs/6962/UMBCBL009_868796962_image_cube_ICA_rgb_0_1_2.png'
rgb_img = np.array(Image.open(image_path))  # shape (H, W, 3)

# rgb_img, density_map = spherical_projection_with_density(df_ball)

print("RGB Image Shape:", rgb_img.shape)
print("RGB Image Dtype:", rgb_img.dtype)
print("RGB Image Max:", rgb_img.max())


df_colored = back_project_color_to_ball(df_ball, rgb_img, inverse_zenith=True)
visualize_point_cloud(df_colored)
