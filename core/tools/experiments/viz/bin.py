import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numba

try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    return args


def read_one_and_viz(file_path):
    print(f"process file: {file_path}")
    points = np.fromfile(file_path.__str__(), dtype=np.float32).reshape(-1, 4)
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3) * 0.3
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    vis.add_geometry(pts)

    vis.run()
    vis.destroy_window()


def main():
    args = parse_config()
    path = Path(args.file)

    def check_valid(file_path):
        return file_path.suffix in ['.bin', 'npy']

    if path.is_dir():
        paths = list(path.iterdir())
        paths.sort()
        paths = [p for p in paths if check_valid(p)]
        for p in paths:
            read_one_and_viz(p)
    else:
        if check_valid(path):
            read_one_and_viz(path)


if __name__ == '__main__':
    main()
