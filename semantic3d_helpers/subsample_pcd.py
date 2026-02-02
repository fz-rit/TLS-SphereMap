import numpy as np
from pathlib import Path
from tools.pcd_utils import log_time_mem_ram, MemoryProfiler
from tools.plot_tools import get_vector_histogram
import time
import pandas as pd
import open3d as o3d
from sklearn.neighbors import KDTree

from collections import Counter
start_time = time.time()
profiler = MemoryProfiler()


def load_pointcloud_and_labels(pcd_path, label_path, dtype_dict):
    column_names = ['X', 'Y', 'Z', 'Intensity', 'r', 'g', 'b']
    df = pd.read_csv(pcd_path, sep='\s+', names=column_names, dtype=dtype_dict)
    pointcloud = df.to_records(index=False)
    labels = np.loadtxt(label_path, dtype=np.int32)
    if len(pointcloud) != len(labels):
        raise ValueError("Mismatch between point count and label count")
    return pointcloud, labels


def classify_rare_and_major_classes(labels, threshold_ratio):
    total_points = len(labels)
    label_counts = Counter(labels)
    rare_classes = [cls for cls, count in label_counts.items() if count / total_points < threshold_ratio]
    major_classes = [cls for cls in label_counts if cls not in rare_classes]
    return rare_classes, major_classes

def stratified_and_uniform_sample(labels, rare_classes, major_classes, keep_num_pts, boost_ratio):
    sample_indices = []
    num_rare_pts = int(keep_num_pts * boost_ratio)
    rare_quota = num_rare_pts // len(rare_classes) if rare_classes else 0

    for cls in rare_classes:
        cls_idx = np.where(labels == cls)[0]
        n = min(len(cls_idx), rare_quota)
        sample_indices.extend(np.random.choice(cls_idx, n, replace=False))

    # Sample remaining from major classes
    remaining_pts = keep_num_pts - len(sample_indices)
    major_idx = np.hstack([np.where(labels == cls)[0] for cls in major_classes])
    sample_indices.extend(np.random.choice(major_idx, remaining_pts, replace=False))

    return np.array(sample_indices)

def save_subsampled_data(points, labels, out_str, savedir, base_stem):
    savedir.mkdir(parents=True, exist_ok=True)
    txt_path = savedir / f"{base_stem}_subsampled_{out_str}.txt"
    labels_path = savedir / f"{base_stem}_subsampled_{out_str}.labels"
    
    np.savetxt(txt_path, np.column_stack([points[field] for field in points.dtype.names]),
               fmt='%.6f %.6f %.6f %d %d %d %d')
    np.savetxt(labels_path, labels, fmt='%d')
    


def hybrid_stratified_downsample(input_pcd_path, 
                                 input_labels_path, 
                                 save_output=True, 
                                 savedir = None,
                                 keep_num_pts=10_000_000,
                                 rare_threshold_ratio=0.01,
                                 rare_class_boost_ratio=0.05,
                                 out_str='hybrid10M'):
    input_pcd_path = Path(input_pcd_path)
    input_labels_path = Path(input_labels_path)
    

    if not input_pcd_path.exists() or not input_labels_path.exists():
        raise FileNotFoundError(f"Missing input: {input_pcd_path} or {input_labels_path}")

    dtype = {'X': np.float32, 'Y': np.float32, 'Z': np.float32,
             'Intensity': np.int64, 'r': np.uint8, 'g': np.uint8, 'b': np.uint8}


    points, labels = load_pointcloud_and_labels(input_pcd_path, input_labels_path, dtype)

    get_vector_histogram(labels, savedir, 
                            f'{input_labels_path.stem}_hist_before_hybrid', 
                            True, log_y=True, visualize=False)

    rare_classes, major_classes = classify_rare_and_major_classes(labels, rare_threshold_ratio)
    indices = stratified_and_uniform_sample(labels, rare_classes, major_classes,
                                            keep_num_pts, rare_class_boost_ratio)

    sampled_points = points[indices]
    sampled_labels = labels[indices]

    get_vector_histogram(sampled_labels, savedir, 
                            f'{input_labels_path.stem}_hist_after_hybrid', 
                            True, log_y=True, visualize=False)

    if save_output:
        save_subsampled_data(sampled_points, sampled_labels, out_str, savedir,
                             input_pcd_path.stem)

    return sampled_points, sampled_labels

def voxel_downsample_points(points, voxel_size):
    """
    Downsample points using Open3D voxel grid and return indices matched to original.
    """
    xyz = np.stack([points['X'], points['Y'], points['Z']], axis=-1)
    rgb = np.stack([points['r'], points['g'], points['b']], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_xyz = np.asarray(down_pcd.points)

    # Match downsampled points to nearest in original to retrieve full info
    from sklearn.neighbors import KDTree
    tree = KDTree(xyz)
    _, idx = tree.query(down_xyz, k=1)

    return idx[:, 0]  # original indices

def voxel_grid_downsample(points,
                          labels,
                          voxel_size=0.05,
                          output_dir='voxel_subsampled'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot histogram before
    get_vector_histogram(labels, output_dir,
                            f"hist_before_voxel_downsample", True, log_y=False, visualize=False)

    # 2. Perform voxel downsampling
    step_time = time.time()
    selected_idx = voxel_downsample_points(points, voxel_size)
    final_points = points[selected_idx]
    final_labels = labels[selected_idx]

    # 3. Histogram after
    get_vector_histogram(final_labels, output_dir,
                            f"hist_after_voxel_downsample", True, log_y=False, visualize=False)

    # 4. Save
    base_name = f"voxel_subsampled_{voxel_size}"
    save_subsampled_data(final_points, final_labels, out_str=base_name,
                         savedir=output_dir, base_stem='')

    return final_points, final_labels


def main():
    # Subsample a large point cloud dataset to 10M points
    # pcd_dir = Path('/home/fzhcis/mylab/data/semantic3d')
    # labels_dir = pcd_dir / 'sem8_labels_training'
    # source_dir = Path('/home/fzhcis/data/semantic3d_full/Semantic3D/train')
    source_dir = Path('/home/fzhcis/data/semantic3d_full/Semantic3D/test')
    output_dir = Path('/shared/rc/mangrove/data/Semantic3D/full-subsampled/test')

    labels_paths = list(source_dir.glob('*.labels'))
    pcd_paths = list(source_dir.glob('*.txt'))
    pcd_paths.sort()
    labels_paths.sort()
    assert len(labels_paths) == len(pcd_paths), "Mismatch in number of labels and point clouds"

    for pcd_path, labels_path in zip(pcd_paths, labels_paths):
        savedir = output_dir / pcd_path.stem
        savedir.mkdir(parents=True, exist_ok=True)
        with profiler.cpu_memory_monitoring() as peak_memory:
            sub_points, sub_labels = hybrid_stratified_downsample(
                pcd_path, 
                labels_path, 
                save_output=True, 
                savedir=savedir,
                keep_num_pts=12_000_000, 
                out_str='12M'
            )
            log_time_mem_ram("Hybrid sampling (stratified + random)", peak_memory, start_time)
            print(f"Subsampled point cloud shape: {len(sub_points)}")
            print(f"Subsampled labels shape: {len(sub_labels)}")
            final_points, final_labels = voxel_grid_downsample(sub_points, sub_labels, voxel_size=0.02, output_dir=savedir)
            print(f"Final voxel grid points shape: {len(final_points)}")
            print(f"Final voxel grid labels shape: {len(final_labels)}")
            log_time_mem_ram("Total execution time", peak_memory, start_time)

if __name__ == "__main__":
    main()

