import open3d as o3d
import numpy as np
import pandas as pd
import time
import torch
import psutil
from pathlib import Path
from matplotlib import pyplot as plt
import threading
from contextlib import contextmanager
from typing import List, Generator
import os

def interactive_visualize_pcd(all_points_xyz: np.ndarray, 
                        all_geom_features: tuple,
                        geom_feature_names: List[str],
                      ) -> None:
    """Visualize curvature and roughness with Open3D.

    Args:
        all_points_xyz (np.ndarray): Array of point cloud coordinates.
        all_geom_features (tuple): Tuple containing curvature and roughness arrays.
        geom_feature_names (List[str]): Names of the geometric features for visualization.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_xyz)
    colormap = plt.get_cmap('plasma')
    for geom_feature, geom_feature_name in zip(all_geom_features, geom_feature_names):
        geom_feature_colors = colormap(geom_feature)[:, :3]  # Convert to RGB colors
        pcd.colors = o3d.utility.Vector3dVector(geom_feature_colors)
        print(f"Displaying {geom_feature_name} visualization...")
        o3d.visualization.draw_geometries([pcd], window_name=f"{geom_feature_name} Visualization")




def export_results(all_points_allinone: pd.DataFrame, 
                   export_path: Path) -> None:
    """Append curvature and roughness to points and export.

    Args:
        all_points_allinone (pd.DataFrame): Point cloud DataFrame with curvature and roughness.
        output_dir (Path): Output directory.
        input_path (Path): Filename for the exported file.
    """
    # str_ls = input_file_stem.split('_')
    # pcd_signature_str = f"pcd_{str_ls[2]}_{str_ls[-1][-4:]}_{out_signature_str}"
    # export_path = output_dir / f"{input_file_stem}_{out_signature_str}.txt"
    all_points_allinone.to_csv(export_path, sep=',', index=False)
    print(f"Exported point cloud with curvature and roughness to {export_path}")


def check_and_clean_for_nans(all_curv_or_rough: np.ndarray) -> np.ndarray:
    """Check for NaN values in curvatures and roughness.

    Args:
        all_curv_or_rough (np.ndarray): Array of curvature values or roughness values.

    Returns:
        np.ndarray: Mask indicating valid values.
    """
    valid_mask = np.ones_like(all_curv_or_rough, dtype=bool)
    if np.isnan(all_curv_or_rough).any():
        print("Warning: NaN values detected in curvature/roughness calculation. Removing NaNs for histogram.")
        valid_mask = ~np.isnan(all_curv_or_rough)
    
    if len(all_curv_or_rough) == 0:
        raise ValueError("Error: No valid curvature or roughness values to plot.")
        
    return valid_mask


class MemoryProfiler:
    """Class for monitoring CPU and GPU memory usage.
    
    Usage:
        profiler = MemoryProfiler()
        with profiler.cpu_memory_monitoring() as peak_memory:
            # Code block to monitor CPU memory usage
            pass
        with profiler.gpu_memory_monitoring():
            # Code block to monitor GPU memory usage
    """

    @contextmanager
    def cpu_memory_monitoring(self) -> Generator[List[int], None, None]:
        """Context manager for monitoring CPU memory usage.

        Yields:
            Generator[List[int], None, None]: Peak memory usage list.
        """
        process = psutil.Process()
        peak_memory = [0]
        stop_event = threading.Event()

        def monitor_memory():
            while not stop_event.is_set():
                mem = process.memory_info().rss
                if mem > peak_memory[0]:
                    peak_memory[0] = mem
                time.sleep(0.1)  # Check every 0.1 seconds

        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()

        try:
            yield peak_memory
        finally:
            stop_event.set()
            monitor_thread.join()

    @contextmanager
    def gpu_memory_monitoring(self) -> Generator[None, None, None]:
        """Context manager for monitoring GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            pass


def create_dir_if_not_exists(directory: Path, ask_user: bool=False) -> None:
    """
    Ask the user whether to create a directory (and its parents) if it does not exist.
    Also display the nearest existing parent directory.

    Args:
        directory (Path): The directory to check and create.
        ask_user (bool): Whether to ask the user for confirmation to create the directory. Default is True.
    
    """
    if directory.exists():
        return

    if not ask_user:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory {directory} created.")
        return

    else:
        # Find the nearest existing parent
        existing_parent = directory
        while not existing_parent.exists():
            existing_parent = existing_parent.parent

        print(f"Directory '{directory}' does not exist.")
        print(f"The nearest existing parent is: '{existing_parent}'")

        # Ask user whether to create
        while True:
            response = input("Do you want to create the missing directory (and any missing parents)? (y/n): ").strip().lower()
            if response == 'y':
                directory.mkdir(parents=True, exist_ok=True)
                print(f"Directory {directory} created.")
                break
            elif response == 'n':
                print(f"Program stopped due to lack of the path '{directory}'.")
                break
            else:
                print("Please enter 'y' or 'n'.")


def log_time_mem_ram(msg="", peak_memory=None, start_time=None):
    """Log the current time, memory, and RAM usage."""
    peak_cpu_memory_mb = peak_memory[0] / (1024 * 1024)
    elapsed_time = time.time() - start_time if start_time is not None else 0
    print(f"Elapsed Time {msg}: {elapsed_time:.2f} seconds")
    print(f"Peak CPU Memory Used {msg}: {peak_cpu_memory_mb:.2f} MB")


def check_nprocs(min_required=4):
    nprocs = len(os.sched_getaffinity(0))
    if nprocs is None:
        raise RuntimeError("Could not determine the number of processors.")
    if nprocs < min_required:
        raise RuntimeError(f"At least {min_required} processors are required, but only {nprocs} available.")
    print(f"Number of processors available: {nprocs}")

if __name__ == "__main__":
    # test_dir = Path("/home/fzhcis/mylab/gdrive/projects_with_Jan/for_Fei/tls_scans_4fei/output")
    # create_dir_if_not_exists(test_dir)
    check_nprocs(min_required=4)