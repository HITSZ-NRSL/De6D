import argparse
import glob
import os
import sys
from pathlib import Path
import cv2

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
from pcdet.utils import calibration_kitti
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.augmentor.augmentor_utils import random_global_make_slope
import imageio
import pickle


class DemoDataset(DatasetTemplate):
    def __init__(self, handle, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.handle = handle

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        calib = calibration_kitti.Calibration("experiments/demo_data/calib.txt")
        img_shape = np.array([375, 1242])
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = self.handle.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]
        params = [[20, 0],
                  [24, 0]]
        _, points, _, _ = random_global_make_slope(
            gt_boxes=np.zeros([0, 9]), points=points, smooth=True,
            params=(params[0][0], params[0][1], *np.deg2rad([params[1][0], params[1][1]])))
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
    parser.add_argument('--tag', type=str, default='model_name')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


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


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------video of SlopedKITTI-------------------------')
    logger.info('----------------load dataset and model-------------------------')
    print(f"args.data_path: {args.data_path}")
    aug_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False
    )

    demo_dataset = DemoDataset(handle=aug_dataset,
                               dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
                               root_path=Path(args.data_path), ext=args.ext, logger=logger
                               )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    logger.info('-----------------  construct windows -------------------------')
    """
    even callback
    """
    need_stop = True
    stop = True

    def key_action_callback(vis, action, mods):
        nonlocal stop
        stop = False

    """
    create window weight
    """
    viz_style = {'pointrcnn': {'bg': [0, 0, 0],
                               'fg_poitns': [1.0, 0.5, 0], 'bg_poitns': [0.8, 0.8, 0.8],
                               'gt_boxes': [1, 0, 0], 'pred_boxes': [0, 1, 0], },
                 'centerpoint': {'bg': [1.0, 1.0, 1.0],
                                 'fg_poitns': [0.0, 0.2, 0.7], 'bg_poitns': [0, 0.7, 1.0],
                                 'gt_boxes': [0, 0.7, 0], 'pred_boxes': [0.7, 0, 0], }, }
    color_map = viz_style['pointrcnn']

    vis = open3d.visualization.VisualizerWithKeyCallback()
    image_size = (1600, 610)
    vis.create_window(width=image_size[0], height=image_size[1])
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array(color_map['bg'])
    vis.register_key_action_callback(32, key_action_callback)

    logger.info('------------------- record video -----------------------------')
    save_dir = Path('experiments/results/gif') / demo_dataset.root_path.parent.parent.stem
    print(f"save_dir: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_save_dir = save_dir / 'pred'
    pred_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info('--------------------- inference -----------------------------')
    image_buff_front = []
    image_buff_side = []
    boxes_buff = []
    cnt = 0
    skip = 3
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            cnt += 1
            if cnt % skip != 1:
                continue

            print(data_dict['file'])
            data_dict.pop('file')
            raw_points = data_dict['raw_points'][:, :3]

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            vis.clear_geometries()
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_boxes = pred_boxes[pred_dicts[0]['pred_labels'] == 1].cpu().numpy()
            if pred_boxes.shape[1] != 9:
                pred_boxes = np.hstack([pred_boxes, np.zeros([pred_boxes.shape[0], 9 - pred_boxes.shape[1]])])
            print(f"points: {raw_points.shape}, pred_boxes: {pred_boxes.shape}")
            if pred_boxes.shape[0] > 0:
                boxes_buff.append(pred_boxes)
                in_box_mask = box_utils.points_in_boxes3d(raw_points, pred_boxes[:, :9])
                in_box_mask = in_box_mask >= 0
                fg_points = raw_points[in_box_mask, :]
                add_keypoint(vis, fg_points, radius=0.08, color=color_map['fg_poitns'])

            # display phase
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            # add_keypoint(vis, fg_points, radius=0.08, color=color_map['fg_poitns'])
            vis = V.draw_box(vis, pred_boxes, color_map['pred_boxes'], width=7)

            camera_parameter_path = 'experiments/viz/viewpoints/vp_gig_1_1600_610.json'
            params = open3d.io.read_pinhole_camera_parameters(camera_parameter_path)
            vc = vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            img = np.asarray(img)
            img = np.uint8(img * 255)
            img = cv2.resize(img, image_size)
            image_buff_front.append(img)

            camera_parameter_path = 'experiments/viz/viewpoints/vp_gig_2_1600_610.json'
            params = open3d.io.read_pinhole_camera_parameters(camera_parameter_path)
            vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            img = np.asarray(img)
            img = np.uint8(img * 255)
            img = cv2.resize(img, image_size)
            image_buff_side.append(img)
            # while stop:
            #     vis.poll_events()
            #     vis.update_renderer()
        imageio.mimsave((save_dir / f'{args.tag}.gif').__str__(), image_buff_front, fps=15)
        imageio.mimsave((save_dir / f'{args.tag}_side.gif').__str__(), image_buff_side, fps=15)
        with open((pred_save_dir / f'{args.tag}_pred').__str__(), 'wb') as f:
            pickle.dump(boxes_buff, f)


if __name__ == '__main__':
    main()
