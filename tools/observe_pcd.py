import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def observe_pcd_class_freq(pcd_path: Path) -> None:
    """
    Observe the point cloud data in a .csv file. Calculate the frequency of each class_id.
    :param pcd_path: Path to the .csv file.
    :return: None
    """

    # Read the .csv file
    pcd = pd.read_csv(pcd_path)

    # Calculate the frequency of each class_id
    class_id_freq = pcd['class_id'].value_counts().to_dict()

    # Print the frequency of each class_id
    print(f'class_id frequency of {pcd_path.stem}:')
    for class_id, freq in sorted(class_id_freq.items(), key=lambda x: x[0]):
            print(f'class_id {class_id}: {freq}')

def observe_histogram(pcd_path: Path) -> None:
    """
    Observe the point cloud data in a .csv file. Calculate the histogram of each class_id.
    :param pcd_path: Path to the .csv file.
    :return: None
    """

    pcd = pd.read_csv(pcd_path)

    col_names = pcd.columns.tolist()
    print(f'col_names of {pcd_path.stem}:') 
    print(col_names)

    plot_rows = len(col_names) // 4 + 1

    fig, axes = plt.subplots(plot_rows, 4, figsize=(8, 4*plot_rows))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, len(col_names)))
    for i, col_name in enumerate(col_names):
        ax = axes[i]
        bins = 2 if col_name == 'Return Number' else 256
        ax.hist(pcd[col_name], bins=bins, color=colors[i], alpha=0.75)
        ax.set_title(col_name)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

if __name__ == '__main__':
    key_str1 = 'U1B375'
    key_str2 = 'UMBCBL009_2024-04-03-21-08-54_U1B375North_060180_000538.800_2095742865'
    file_dir = Path(f'/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/palau_2024/{key_str1}/{key_str2}/outputs/pcd')
    input_path = file_dir / f'{key_str2}_color.csv'
    # observe_pcd_class_freq(input_path)
    observe_histogram(input_path)
    plt.show()