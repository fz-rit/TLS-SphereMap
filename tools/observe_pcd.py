import pandas as pd
import numpy as np
from pathlib import Path


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

if __name__ == '__main__':
    key_str1 = 'ALRSET2'
    key_str2 = 'UMBCBL009_2024-03-28-02-23-06_ALRSET22_060180_000329.800_1303851536'
    file_dir = Path(f'/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/palau_2024/{key_str1}/{key_str2}/outputs')
    input_path = file_dir / f'{key_str2}_seg.csv'
    observe_pcd_class_freq(input_path)