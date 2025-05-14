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
from open3d.visualization import rendering


def interactive_visualize_pcd(all_points_xyz: np.ndarray, 
                      all_curvatures: np.ndarray, 
                      all_roughness: np.ndarray,
                      neighbor_radius: float,
                      ) -> None:
    """Visualize curvature and roughness with Open3D.

    Args:
        all_points_xyz (np.ndarray): Array of points_xyz.
        all_curvatures (np.ndarray): Array of curvature values.
        all_roughness (np.ndarray): Array of roughness values.
        neighbor_radius (float): Radius for curvature & roughness estimation.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_xyz)
    colormap = plt.get_cmap('plasma')
    all_curvatures_colors = colormap(all_curvatures)[:, :3]
    all_roughness_colors = colormap(all_roughness)[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(all_curvatures_colors)
    print("Displaying curvature visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Curvature Visualization (r={neighbor_radius})")

    pcd.colors = o3d.utility.Vector3dVector(all_roughness_colors)
    print("Displaying roughness visualization...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Roughness Visualization (r={neighbor_radius})")




def export_results(all_points_allinone: pd.DataFrame, 
                   neighbor_radius: float, 
                   output_dir: Path, 
                   input_file_stem: str) -> None:
    """Append curvature and roughness to points and export.

    Args:
        all_points_allinone (pd.DataFrame): Point cloud DataFrame with curvature and roughness.
        output_dir (Path): Output directory.
        input_path (Path): Filename for the exported file.
    """

    export_path = output_dir / f"{input_file_stem}_ncr_{neighbor_radius:.2f}.txt"
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


def create_dir_if_not_exists(directory: Path, ask_user: bool=True) -> None:
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


if __name__ == "__main__":
    test_dir = Path("/home/fzhcis/mylab/gdrive/projects_with_Jan/for_Fei/tls_scans_4fei/output")
    create_dir_if_not_exists(test_dir)