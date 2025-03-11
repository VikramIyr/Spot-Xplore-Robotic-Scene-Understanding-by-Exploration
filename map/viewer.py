import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from plyfile import PlyData

def main():
    # Read the PLY file using plyfile
    plydata = PlyData.read("graphnav_pcl.ply")
    vertex_data = plydata['vertex'].data

    # Extract x, y, z coordinates
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    # Use intensity data if available
    if 'intensity' in vertex_data.dtype.names:
        intensity = vertex_data['intensity'].astype(np.float64)
        # Normalize intensity to the range [0, 1]
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # Map normalized intensity to colors using the 'jet' colormap (an RGB style colormap)
        colors = cm.jet(intensity_norm)[:, :3]
    else:
        # If no intensity, fallback to using RGB if available
        if all(key in vertex_data.dtype.names for key in ['red', 'green', 'blue']):
            colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T / 255.0
        else:
            # Otherwise, assign a default gray color
            colors = np.tile([0.1, 0.1, 0.7], (points.shape[0], 1))

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                      window_name="RGB Colored Point Cloud",
                                      width=800,
                                      height=600)

if __name__ == "__main__":
    main()
