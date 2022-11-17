import argparse
import glob
import os
import sys
from pathlib import Path
import cv2
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
from pcdet.utils import calibration_kitti
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
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

        points = points[points[:, 0] > 0]

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


def point_cloud_random_make_slope(gt_boxes=None, gt_points=None, params=None, rotate_point=None, rotate_angle=None,
                                  smooth=False):
    from pcdet.utils import common_utils
    from scipy.spatial.transform import Rotation
    def random(n=1):
        return (np.random.random(n) - 0.5) * 2

    ##########################################
    # get rotate point and rotate angle
    ##########################################
    assert params is not None

    dist_mean, dist_var, angle_mean, angle_var = params
    points = gt_points.copy()

    if rotate_point is None:
        mean, var = np.array([dist_mean, 0]), np.array([dist_var, 0])
        polar_pos = mean + random(2) * var
        rotate_point = np.array([polar_pos[0] * np.cos(polar_pos[1]), polar_pos[0] * np.sin(polar_pos[1]), 0])

    x0, y0 = rotate_point[0], rotate_point[1]
    if rotate_angle is None:
        mean, var = angle_mean, angle_var
        k0 = y0 / x0
        k1 = -1 / (k0 + 1e-6)
        v = np.array([x0 - 0, y0 - (-x0 * k1 + y0), 0])
        v /= np.linalg.norm(v)
        angle = mean + random() * var
        v *= angle
        direction = np.sign(np.cross(rotate_point, v)[2])
        # v *= -1 if direction > 0 else 1
        rotate_angle = v

    ##########################################
    # apply sloped-aug in smooth condition
    ##########################################
    if smooth:
        radius, bins = rotate_point[0] / np.abs(rotate_angle[1]), 8
        alpha = rotate_angle[1]
        dist = rotate_point[0]
        for theta in np.linspace(0, alpha, bins):
            delta = alpha / bins
            center = np.array([dist, 0, radius])
            rotate_point = center + np.array([-radius * np.sin(theta), 0, -radius * np.cos(theta)])
            rotate_angle = np.array([0, delta, 0])
            gt_boxes, points, rotate_point, rotate_angle = point_cloud_random_make_slope(
                gt_boxes, points,
                params=params,
                rotate_angle=rotate_angle,
                rotate_point=rotate_point)
    else:
        k = rotate_angle[1] / (rotate_angle[0] + 1e-6)
        sign = np.sign(k * (0 - x0) + y0 - 0)
        in_plane_mask = np.sign(k * (points[:, 0] - x0) + y0 - points[:, 1]) != sign
        slope_points = points[in_plane_mask]
        slope_points[:, 0:3] -= rotate_point
        rot = Rotation.from_rotvec(rotate_angle).as_matrix()
        slope_points[:, 0:3] = (slope_points[:, 0:3].dot(rot.T))
        slope_points[:, 0:3] += rotate_point
        points[in_plane_mask] = slope_points
        if gt_boxes is not None:
            # gt_box
            if gt_boxes.shape[1] < 9:
                gt_boxes = np.concatenate((gt_boxes, np.zeros([gt_boxes.shape[0], 2])), axis=1)
            in_plane_mask = np.sign(k * (gt_boxes[:, 0] - x0) + y0 - gt_boxes[:, 1]) != sign  # box position mask
            slope_box = gt_boxes[in_plane_mask]
            slope_box[:, :3] -= rotate_point
            slope_box[:, :3] = (slope_box[:, :3].dot(rot.T))
            slope_box[:, :3] += rotate_point
            gt_boxes[in_plane_mask] = slope_box
            euler = Rotation.from_rotvec(rotate_angle).as_euler('XYZ')
            # pts × ((RxRy)Rz).T
            # boxes(9)[x, y, z, dx, dy, dz, rz(, ry, rx)[Rot=RxRyRz]]
            gt_boxes[in_plane_mask, 7] += euler[1]  # y
            gt_boxes[in_plane_mask, 8] += euler[0]  # y
            gt_boxes[:, 6:9] = common_utils.limit_period(
                gt_boxes[:, 6:9], offset=0.5, period=2 * np.pi
            )
    return gt_boxes, points, rotate_point, rotate_angle


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
    # vis.create_window(width=1224, height=595)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array(color_map['bg'])
    # vis.register_key_action_callback(32, key_action_callback)

    logger.info('------------------- record video -----------------------------')
    save_dir = Path('experiments/results/gif') / demo_dataset.root_path.parent.parent.stem
    print(f"save_dir: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_save_dir = save_dir / 'pred'
    pred_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info('--------------------- inference -----------------------------')
    image_buff_front = []
    boxes_buff = []
    cnt = 0
    skip = 3
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            cnt += 1
            if cnt % skip != 1:
                continue

            print(data_dict['file'])
            vis.clear_geometries()
            vis.poll_events()
            vis.update_renderer()

            data_dict.pop('file')
            raw_points = data_dict['raw_points'][:, :3]
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_boxes = pred_boxes[pred_dicts[0]['pred_labels'] == 1].cpu().numpy()
            pred_boxes = pred_boxes[np.linalg.norm(pred_boxes[:, 0:3], axis=-1) > 2.5]
            if pred_boxes.shape[1] != 9:
                pred_boxes = np.hstack([pred_boxes, np.zeros([pred_boxes.shape[0], 9 - pred_boxes.shape[1]])])
            print(f"points: {raw_points.shape}, pred_boxes: {pred_boxes.shape}")
            if pred_boxes.shape[0] > 0:
                boxes_buff.append(pred_boxes)
                in_box_mask = box_utils.points_in_boxes3d(raw_points, pred_boxes[:, :9])
                in_box_mask = in_box_mask >= 0
                fg_points = raw_points[in_box_mask, :]
                print(f"fg points number {fg_points.shape[0]}")
                print(f"add keypoints ...")
                if fg_points.shape[0] > 7000:
                    flags = np.random.choice(fg_points.shape[0], 7000, replace=False)
                    fg_points = fg_points[flags]
                add_keypoint(vis, fg_points, radius=0.08, color=color_map['fg_poitns'])
            # display phase
            print(f"add cloud ...")
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            V.draw_box(vis, pred_boxes, color_map['pred_boxes'], width=7)

            camera_parameter_path = 'experiments/viz/viewpoints/vp_gz_fig_1.json'
            params = open3d.io.read_pinhole_camera_parameters(camera_parameter_path)
            vc = vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            img = np.asarray(img)
            img = np.uint8(img * 255)
            img = cv2.resize(img, image_size)
            image_buff_front.append(img)


        imageio.mimsave((save_dir / f'{args.tag}.gif').__str__(), image_buff_front, fps=15)
        with open((pred_save_dir / f'{args.tag}_pred').__str__(), 'wb') as f:
            pickle.dump(boxes_buff, f)



if __name__ == '__main__':
    main()
