import argparse
import glob
import os
import sys
from pathlib import Path
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
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=40)
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


from scipy.spatial.transform import Rotation as Rot


def vec2rot(vec):
    v = vec / np.linalg.norm(vec)
    u = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, v)
    sin_a = np.linalg.norm(axis)
    axis /= sin_a
    cos_a = np.dot(u, v)
    alpha = np.arctan2(sin_a, cos_a)
    rot1 = Rot.from_rotvec(axis * alpha).as_matrix()
    return rot1


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
        if action == 0:
            nonlocal stop
            stop = False

    """
    create window weight
    """
    viz_style = {'pointrcnn': {'bg': [1, 1, 1],
                               'fg_poitns': [1.0, 0.5, 0], 'bg_poitns': [0.5, 0.5, 0.5],
                               'gt_boxes': [1, 0, 0], 'pred_boxes': [0, 1, 0], },
                 'centerpoint': {'bg': [1.0, 1.0, 1.0],
                                 'fg_poitns': [0.0, 0.2, 0.7], 'bg_poitns': [0, 0.7, 1.0],
                                 'gt_boxes': [0, 0.7, 0], 'pred_boxes': [0.7, 0, 0], }, }
    color_map = viz_style['pointrcnn']

    vis = open3d.visualization.VisualizerWithKeyCallback()
    image_size = (1920, 1080)
    vis.create_window(width=image_size[0], height=image_size[1])
    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.array(color_map['bg'])
    vis.register_key_action_callback(32, key_action_callback)

    logger.info('--------------------- inference -----------------------------')
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print(data_dict['file'])
            data_dict.pop('file')
            raw_points = data_dict['raw_points'][:, :3]

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            vis.clear_geometries()
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_boxes = pred_boxes[pred_dicts[0]['pred_labels'] == 1].cpu().numpy()
            sampled_list = data_dict['point_coords_list']

            def get_box_nearby(boxes: np.ndarray, coord: np.ndarray or list):
                return np.where(np.linalg.norm(boxes[:, :3] - coord[None, ...], axis=-1) < 1)[0][0]

            def point_select(points, c, r):
                size = np.array([-6, -6, -3,
                                 6, 2.1, 3])
                points -= c
                valid_flag1 = np.logical_and(points[:, 0] > size[0], points[:, 0] < size[0 + 3])
                valid_flag2 = np.logical_and(points[:, 1] > size[1], points[:, 1] < size[1 + 3])
                valid_flag3 = np.logical_and(points[:, 2] > size[2], points[:, 2] < size[2 + 3])
                valid_flag = np.logical_and(valid_flag1, np.logical_and(valid_flag2, valid_flag3))
                radius_flag = np.linalg.norm(points[:, :3], axis=-1) < radius
                valid_flag = np.logical_and(valid_flag, radius_flag)
                points = points[valid_flag]
                return points, valid_flag

            def view_and_capture(file_name, camera_parameter_path):
                params = open3d.io.read_pinhole_camera_parameters(camera_parameter_path)
                vc = vis.get_view_control()
                vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                vis.update_renderer()
                print(f"capture [{file_name}] ...")
                img = vis.capture_screen_float_buffer(do_render=True)
                img = np.asarray(img)
                file_path = Path(file_name)
                file_path.parent.mkdir(exist_ok=True, parents=True)
                plt.imsave(file_name, img)

            def wait():
                nonlocal stop
                stop = True
                while stop:
                    vis.poll_events()
                    vis.update_renderer()

            ############################################################################################################
            # data preparing
            ############################################################################################################
            reserve = [get_box_nearby(pred_boxes, np.array([13.74, 3.02, -0.784])),
                       get_box_nearby(pred_boxes, np.array([18.02, 2.53, 0]))]
            print(reserve)
            pred_boxes = pred_boxes[reserve, :]
            center = np.mean(pred_boxes[:, :3], axis=0)
            print(center)
            pred_boxes[:, :3] -= center

            # points selected
            radius = 7
            raw_points, _ = point_select(raw_points, center, radius)
            points_4096, _ = point_select(sampled_list[0].cpu().numpy()[:, 1:], center, radius)
            points_1024, _ = point_select(sampled_list[1].cpu().numpy()[:, 1:], center, radius)
            points_512, _ = point_select(sampled_list[2].cpu().numpy()[:, 1:], center, radius)
            points_256, vote_ind = point_select(data_dict['point_candidate_coords'].cpu().numpy()[:, 1:], center,
                                                radius)
            points_vote, _ = point_select(data_dict['point_vote_coords'].cpu().numpy()[vote_ind, 1:], center, radius)

            ############################################################################################################
            # point cloud drawing
            ############################################################################################################
            add_keypoint(vis, raw_points, radius=0.03, color=color_map['bg_poitns'])
            view_and_capture("experiments/results/pipeline/input.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # 4096 downsampled points drawing
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            add_keypoint(vis, points_4096, radius=0.04, color=[0.35, 0.35, 0.35])
            view_and_capture("experiments/results/pipeline/points_4096.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')

            wait()
            ############################################################################################################
            # 1024 downsampled points drawing
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            add_keypoint(vis, points_1024, radius=0.05, color=[0.25, 0.25, 0.25])
            view_and_capture("experiments/results/pipeline/points_1024.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # 512 downsampled points drawing
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            add_keypoint(vis, points_512, radius=0.055, color=[0.15, 0.15, 0.15])
            view_and_capture("experiments/results/pipeline/points_512.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # 256 downsampled points drawing
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            add_keypoint(vis, points_256, radius=0.06, color=[0.05, 0.05, 0.05])
            view_and_capture("experiments/results/pipeline/points_256.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw offset prediction
            ############################################################################################################
            points_offset = data_dict['vote_offsets'].permute(0, 2, 1).cpu().numpy()[0, vote_ind, :]
            for i in range(points_256.shape[0]):
                p1 = points_256[i]
                p2 = p1 + points_offset[i, :]
                center = (p1 + p2) / 2
                direction = p1 - p2
                length = np.linalg.norm(direction)
                v1 = np.array([0, 0, 1])
                v2 = direction / length
                v3 = np.cross(v1, v2)
                sinx = np.linalg.norm(v3)
                v3 = v3 / sinx
                cosx = np.inner(v1, v2)
                theta = np.arctan2(sinx, cosx)
                rot = Rot.from_rotvec(theta * v3)
                cylinder = open3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=length)
                cylinder.compute_vertex_normals()
                cylinder.translate(center)
                cylinder.rotate(rot.as_matrix())
                cylinder.paint_uniform_color([1, 0, 0])
                vis.add_geometry(cylinder)
            view_and_capture("experiments/results/pipeline/vote_offset.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw center prediction
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            add_keypoint(vis, points_vote, radius=0.08, color=[0.9, 0.7, 0.0])
            view_and_capture("experiments/results/pipeline/points_vote.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw ground segmentation
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            in_box_mask = box_utils.points_in_boxes3d(points_vote, pred_boxes)
            for i in range(pred_boxes.shape[0]):
                flags = in_box_mask == i
                if pred_boxes[i, 7] < 0:
                    add_keypoint(vis, points_vote[flags], color=[0.8, 0.1, 0.1], radius=0.10)
                else:
                    add_keypoint(vis, points_vote[flags], color=[0.1, 0.1, 0.8], radius=0.10)
            view_and_capture("experiments/results/pipeline/points_seg.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()

            ############################################################################################################
            # draw head-offset
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            in_box_mask = box_utils.points_in_boxes3d(points_vote, pred_boxes)
            for i in range(pred_boxes.shape[0]):
                flags = in_box_mask == i
                if pred_boxes[i, 7] < 0:
                    add_keypoint(vis, points_vote[flags], color=[0.8, 0.1, 0.1], radius=0.05)
                else:
                    add_keypoint(vis, points_vote[flags], color=[0.1, 0.1, 0.8], radius=0.05)
            box_encodings = data_dict['point_reg_preds'].cpu().numpy()[vote_ind, :]
            box_offsets, box_rot_code, *cgs = np.split(box_encodings, [6, 12 * 2 + 2], axis=-1)
            offset, size = np.split(box_offsets, [3], axis=-1)
            print(f"offset: {offset.shape}, size: {size.shape}")
            for i in range(points_vote.shape[0]):
                p1 = points_vote[i]
                p2 = p1 + offset[i, :]
                center = (p1 + p2) / 2
                direction = p1 - p2
                length = np.linalg.norm(direction)
                v1 = np.array([0, 0, 1])
                v2 = direction / length
                v3 = np.cross(v1, v2)
                sinx = np.linalg.norm(v3)
                v3 = v3 / sinx
                cosx = np.inner(v1, v2)
                theta = np.arctan2(sinx, cosx)
                rot = Rot.from_rotvec(theta * v3)
                cylinder = open3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=length)
                cylinder.compute_vertex_normals()
                cylinder.translate(center)
                cylinder.rotate(rot.as_matrix())
                cylinder.paint_uniform_color([0, 1, 0])
                vis.add_geometry(cylinder)

            view_and_capture("experiments/results/pipeline/head_offset.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw head-dimension
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            box_encodings = data_dict['point_reg_preds'].cpu().numpy()[vote_ind, :]
            box_offsets, box_rot_code, *cgs = np.split(box_encodings, [6, 12 * 2 + 2], axis=-1)
            offset, size = np.split(box_offsets, [3], axis=-1)
            size = np.exp(size)
            boxes = np.concatenate((points_vote, size, np.zeros_like(size)[:, 0, None]), axis=-1)
            V.draw_box(vis, boxes, [0, 0, 0])
            view_and_capture("experiments/results/pipeline/head_dimension.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw head-orientation
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            print(pred_boxes[:, 6:9])
            for i in range(pred_boxes.shape[0]):
                coord = pred_boxes[i, :3]
                ypr = pred_boxes[i, 6:]
                axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=coord)
                axis.rotate(Rot.from_euler('zyx', ypr).as_matrix())
                vis.add_geometry(axis)
            view_and_capture("experiments/results/pipeline/head_orientation.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw head-cls
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            scores = data_dict['point_cls_scores'].cpu().numpy()[vote_ind]
            for i in range(points_vote.shape[0]):
                add_keypoint(vis, points_vote[i][None, :], color=[scores[i], 0, 0], radius=0.08)
            view_and_capture("experiments/results/pipeline/head_classification.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()
            ############################################################################################################
            # draw final prediction
            ############################################################################################################
            vis.clear_geometries()
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(raw_points[:, :3])
            cloud.colors = open3d.utility.Vector3dVector(np.ones_like(raw_points))
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            in_box_mask = box_utils.points_in_boxes3d(raw_points, pred_boxes[:, :9])
            in_box_mask = in_box_mask >= 0
            fg_points = raw_points[in_box_mask, :]
            add_keypoint(vis, fg_points, radius=0.08, color=color_map['fg_poitns'])
            add_cloud(vis, raw_points, color=color_map['bg_poitns'])
            vis = V.draw_box(vis, pred_boxes, color_map['pred_boxes'], width=7)
            view_and_capture("experiments/results/pipeline/final_prediction.png",
                             'experiments/viz/viewpoints/video_pipeline_vp2.json')
            wait()


if __name__ == '__main__':
    main()
