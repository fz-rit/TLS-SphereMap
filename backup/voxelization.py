import open3d as o3d
from pathlib import Path

# Load the .pcd file
root_dir = Path('/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/harvard_forest_33/merged')
pc_path = root_dir / '3301_02_03_merged.pcd'
voxel_size = 0.1
voxel_center_out_path = root_dir / f'voxel_center_{voxel_size}.pcd'
pcd = o3d.io.read_point_cloud(str(pc_path))
o3d.visualization.draw_geometries([pcd], window_name="Raw Point Cloud")

# Voxelize the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
o3d.visualization.draw_geometries([voxel_grid], window_name=f"Voxelized Point Cloud ({voxel_size})")

# Save the voxelized point cloud as voxel centers
voxel_centers = voxel_grid.get_voxels()
center_points = [voxel.grid_index * voxel_size for voxel in voxel_centers]
voxel_pcd = o3d.geometry.PointCloud()
voxel_pcd.points = o3d.utility.Vector3dVector(center_points)
o3d.io.write_point_cloud(str(voxel_center_out_path), voxel_pcd)
