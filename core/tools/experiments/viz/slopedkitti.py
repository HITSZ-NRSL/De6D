import argparse
import glob
from pathlib import Path

try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.slopedkitti.kitti_dataset import SlopedKittiDataset
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import box_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--rand', type=int, default=None)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    dataset = SlopedKittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False
    )
    print('dataset.root_path: %s' % dataset.root_path)
    print('dataset.root_split_path: %s' % dataset.root_split_path)
    print('dataset.sample_id_list: %d' % dataset.sample_id_list.__len__())

    index_list = np.arange(0, len(dataset)) if args.rand is None else np.random.randint(0, len(dataset), size=args.rand)

    for index in index_list:
        i = index
        data_dict = dataset[index]
        frame_id = data_dict['frame_id']
        points = data_dict['points']
        boxes = data_dict['gt_boxes']
        slope_data = dataset.get_slope_plane(frame_id)
        print(i, frame_id)

        draw_points = points
        draw_boxes = boxes
        ## viz
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.ones(3) * 0.3
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(draw_points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.zeros((draw_points.shape[0], 3)))
        vis.add_geometry(pts)

        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        mesh_sphere.translate(slope_data[0, :])
        mesh_sphere.compute_vertex_normals()
        vis.add_geometry(mesh_sphere)

        vis = V.draw_box(vis, draw_boxes, (0, 0, 1))

        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    main()
