import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("/Users/keremkilic/Desktop/3D Vision/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/map/graphnav_pcl.ply")

# **Number of points**
num_points = len(pcd.points)
print(f"Number of Points: {num_points}")

# **Centroid of the point cloud**
points = np.asarray(pcd.points)
centroid = np.mean(points, axis=0)
print(f"Centroid: {centroid}")

# **Bounding boxes**
aabb = pcd.get_axis_aligned_bounding_box()
obb = pcd.get_oriented_bounding_box()
print(f"AABB Min: {aabb.min_bound}, Max: {aabb.max_bound}")
print(f"OBB Center: {obb.center}")

# **Point Cloud Density (Mean Distance Between Points)**
from scipy.spatial import cKDTree

kdtree = cKDTree(points)
distances, _ = kdtree.query(points, k=2)  # k=2 because the first neighbor is the point itself
mean_distance = np.mean(distances[:, 1])  # Ignore first column (distance to itself)
print(f"Mean Point Cloud Density: {mean_distance}")

# **Eigenvalues of the Covariance Matrix (Shape Descriptor)**
cov_matrix = np.cov(points.T)  # Compute covariance matrix
eigenvalues, _ = np.linalg.eig(cov_matrix)
print(f"Eigenvalues of Covariance Matrix: {eigenvalues}")

# **Visualize Bounding Boxes**
o3d.visualization.draw_geometries([pcd, aabb], window_name="Axis-Aligned Bounding Box")
o3d.visualization.draw_geometries([pcd, obb], window_name="Oriented Bounding Box")
