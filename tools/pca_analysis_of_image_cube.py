

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
from processing.spherical_projection import load_image_cube_and_metadata
from tools.pca_helper import compute_band_correlation, compute_pca_components, compute_mnf, compute_ica
from tools.plot_tools import plot_correlation_matrix, plot_pca_components, plot_rgb_permutations
import matplotlib.pyplot as plt
from tools.config_loader import CONFIG
from tools.pcd_utils import create_dir_if_not_exists


output_dir = Path(CONFIG['global']['output_dir'])
input_file_stem = CONFIG['global']['input_file_stem']
key_str = input_file_stem.split('_')[0] + '_' + input_file_stem.split('_')[-1]
image_cube_path = output_dir / 'img' / f'{key_str}_image_cube.npy'

output_stem = image_cube_path.stem
image_cube, metadata = load_image_cube_and_metadata(image_cube_path)
image_cube = image_cube.astype(np.float32) # (H, W, C)


band_names = ['Intensity', 
            'Z Map Inverse', 
            'Range', 
            'Rn',
            'Gn',
            'Bn']


corr_matrix = compute_band_correlation(image_cube)
save_dir = output_dir / 'pca'
create_dir_if_not_exists(save_dir)
plot_correlation_matrix(corr_matrix, band_names = band_names, output_dir=save_dir, output_stem=output_stem)


pcs, pca = compute_pca_components(image_cube, n_components=3)
mnf_components = compute_mnf(image_cube, n_components=3)
ica_components = compute_ica(image_cube, n_components=3)

for components, name in zip([pcs, mnf_components, ica_components], ['PCA', 'MNF', 'ICA']):
    out_file = f"{output_stem}_{name}"
    plot_pca_components(components, save_dir, output_stem=out_file)
    plot_rgb_permutations(components, save_dir, output_stem=out_file)

plt.show()