import argparse
import glob
import os
import sys
from pathlib import Path

import easydict
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt

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

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            # points = points[::2]

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        raw_points = points.copy()
        input_dict = {
            'file': self.sample_file_list[index],
            'points': points,
            'raw_points': raw_points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--style', type=str, default='pointrcnn', help='visualization style')
    parser.add_argument('--pause', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def add_keypoint(vis, points, radius=0.05, color=None, n=None):
    for i in range(points.shape[0]):
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.translate(points[i])
        mesh_sphere.compute_vertex_normals()
        if color is None:
            mesh_sphere.paint_uniform_color([0.2, 0.2, 0.2])
        else:
            mesh_sphere.paint_uniform_color(np.clip(color, 0, 1))
        if n is None:
            vis.add_geometry(mesh_sphere)
        else:
            vis.add_geometry('sphere' + '_' + str(n) + '_' + str(i), mesh_sphere)


def add_cloud(vis, points, color=None, n=None):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points[:, :3])
    if color is None:
        cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
    else:
        color = np.array(color).reshape(1, 3).repeat(points.shape[0], 0)
        cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))

    if n is None:
        vis.add_geometry(cloud)
    else:
        vis.add_geometry('cloud' + '_' + str(n), cloud)
    return cloud


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
        training=False
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    data_list = [30]

    viz_style = {'pointrcnn': {'bg': [1.0, 1.0, 1.0],
                               'fg_poitns': [1.0, 0.5, 0.0], 'bg_poitns': [0.5, 0.5, 0.5],
                               'gt_boxes': [0, 0, 0], 'pred_boxes': [0, 1, 0], },
                 'centerpoint': {'bg': [1.0, 1.0, 1.0],
                                 'fg_poitns': [0.0, 0.2, 0.7], 'bg_poitns': [0, 0.7, 1.0],
                                 'gt_boxes': [0, 0.7, 0], 'pred_boxes': [0.7, 0, 0], }, }

    color_map = viz_style[args.style]

    """
    even callback
    """
    print(args.pause, args.save)
    need_stop = args.pause
    stop = True

    def key_action_callback(vis, action, mods):
        if action == 0:
            nonlocal stop
            stop = False

    def animation_callback(vis):
        pass

    """
    create window weight
    """
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1575, height=587)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array(color_map['bg'])

    vis.register_key_action_callback(32, key_action_callback)
    params = open3d.io.read_pinhole_camera_parameters("experiments/viz/viewpoints/cover_front.json")

    with torch.no_grad():
        for i, sample_idx in enumerate(data_list):
            print("---------------------------")
            data_dict = demo_dataset[sample_idx]
            frame_id = data_dict['frame_id']
            file_name = data_dict['file']
            data_dict.pop('file')
            print(f"new frame\n{file_name} -> {sample_idx} -> {frame_id}")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            print(f"inference ...")
            (preds1, *_), _ = model.forward(data_dict)

            print(f"get prediction")
            points = data_dict['raw_points'][0].cpu().numpy()
            pred_boxes = preds1['pred_boxes']
            pred_boxes = pred_boxes[preds1['pred_labels'] == 1].cpu().numpy()

            pred_boxes = np.load("experiments/results/Det6D_failure_case_1.npy")
            print(f"points: {points.shape}, pred_boxes: {pred_boxes.shape}")
            if args.load is not None:
                gt_boxes = np.load(args.load)
                zeros = np.zeros_like(gt_boxes)[:, 0:2]
                gt_boxes = np.hstack((gt_boxes, zeros))
                print(f"points: {points.shape}, gt_boxes: {gt_boxes.shape}")
                in_box_mask = box_utils.points_in_boxes3d(points, gt_boxes) >= 0
                gt_points = points[in_box_mask, :][:, 0:3]

            # prediction
            points_draw = points
            pred_boxes_enlarge = pred_boxes.copy()
            in_box_mask = box_utils.points_in_boxes3d(points, pred_boxes_enlarge) >= 0
            fg_points = points[in_box_mask, :][:, 0:3]

            add_cloud(vis, points_draw, color=color_map['bg_poitns'])
            ############################################
            add_keypoint(vis, fg_points, radius=0.082, color=color_map['fg_poitns'])
            add_keypoint(vis, pred_boxes[:, :3], radius=0.12, color=color_map['pred_boxes'])
            vis = V.draw_box(vis, pred_boxes, color_map['pred_boxes'], width=5)
            if args.load is not None:
                add_keypoint(vis, gt_points, radius=0.08, color=color_map['fg_poitns'])
                add_keypoint(vis, gt_boxes[:, :3], radius=0.12, color=color_map['gt_boxes'])
                vis = V.draw_box(vis, gt_boxes, color=color_map['gt_boxes'], width=5)
            ############################################
            if args.save:
                np.save(f"experiments/results/{cfg.MODEL.NAME}", pred_boxes)
            stop = True if need_stop else False

            while stop:
                vc = vis.get_view_control()
                vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                vis.poll_events()
                vis.update_renderer()

            print("clear")
            vis.clear_geometries()

    vis.destroy_window()


if __name__ == '__main__':
    main()
