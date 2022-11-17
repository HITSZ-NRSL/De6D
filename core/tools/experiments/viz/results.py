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
from pcdet.datasets.slopedkitti.kitti_dataset import SlopedKittiDataset


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--style', type=str, default='pointrcnn', help='visualization style')
    parser.add_argument('--pause', action='store_true')

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
    demo_dataset = SlopedKittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # selected for result viz
    data_list = [26, 2360, 166, 1782, 1326, 3328, 67, 96]

    # or
    # data_list = np.random.randint(0, len(demo_dataset), 10)

    viz_style = {'pointrcnn': {'bg': [0, 0, 0],
                               'fg_poitns': [1.0, 0.5, 0], 'bg_poitns': [0.5, 0.5, 0.5],
                               'gt_boxes': [1, 0, 0], 'pred_boxes': [0, 1, 0], },
                 'centerpoint': {'bg': [1.0, 1.0, 1.0],
                                 'fg_poitns': [0.0, 0.2, 0.7], 'bg_poitns': [0, 0.7, 1.0],
                                 'gt_boxes': [0, 0.7, 0], 'pred_boxes': [0.7, 0, 0], }, }

    color_map = viz_style[args.style]

    """
    even callback
    """
    print(args.pause)
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
    vis.create_window(width=1902, height=949)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array(color_map['bg'])

    vis.register_key_action_callback(32, key_action_callback)

    with torch.no_grad():
        for i, sample_idx in enumerate(data_list):
            print("---------------------------")
            data_dict = demo_dataset[sample_idx]
            frame_id = data_dict['frame_id']
            print(f"new frame\n{i} -> {sample_idx} -> {frame_id}")
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            print(f"inference ...")
            pred_dicts, _ = model.forward(data_dict)

            print(f"get prediction")
            points = torch.from_numpy(demo_dataset.get_lidar(data_dict['frame_id'][0])[:, :3])
            calib = demo_dataset.get_calib(data_dict['frame_id'][0])
            img_shape = demo_dataset.kitti_infos[i]['image']['image_shape']
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = demo_dataset.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

            gt_boxes = data_dict['gt_boxes'][0]
            gt_boxes = gt_boxes[gt_boxes[:, 9] == 1]
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_boxes = pred_boxes[pred_dicts[0]['pred_labels'] == 1]

            print(f"points: {points.shape}, gt_boxes: {gt_boxes.shape}, pred_boxes: {pred_boxes.shape}")

            # add cloud
            in_box_mask = box_utils.points_in_boxes3d(points, gt_boxes[:, :9])
            in_box_mask = in_box_mask >= 0
            bg_points = points[in_box_mask == 0, :].cpu().numpy()
            fg_points = points[in_box_mask, :].cpu().numpy()
            add_cloud(vis, bg_points, color=color_map['bg_poitns'])
            add_keypoint(vis, fg_points, radius=0.08, color=color_map['fg_poitns'])

            # add box
            vis = V.draw_box(vis, pred_boxes.cpu().numpy(), color_map['pred_boxes'], width=7)
            vis = V.draw_box(vis, gt_boxes.cpu().numpy(), color_map['gt_boxes'], width=5)

            # capture image
            print("capture ...")

            def view_and_capture(model_name, idx, frame_id, i, camera_parameter_path):
                params = open3d.io.read_pinhole_camera_parameters(camera_parameter_path)
                vc = vis.get_view_control()
                vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                vis.update_renderer()
                img = vis.capture_screen_float_buffer(do_render=True)
                img = np.asarray(img)
                if i == 1:
                    slice_range = [int(img.shape[0] * 4 / 10), int(img.shape[0] / 3 * 2)]
                    img = img[slice_range[0]:slice_range[1], :, :]
                else:
                    slice_range = [int(img.shape[0] * 2 / 10), int(img.shape[0] * 8 / 10)]
                    img = img[slice_range[0]:slice_range[1], :, :]
                output_dir = Path('experiments/results/slopedkitti')
                file_name = f"{model_name}_{idx}_{frame_id}_{i}.png"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / file_name
                plt.imsave(output_path.__str__(), img)

            vis.get_render_option().point_size = 3.5
            view_and_capture(cfg.MODEL.NAME, sample_idx, frame_id, 0,
                             "experiments/viz/viewpoints/camera2_frontview2.json")
            vis.get_render_option().point_size = 2.0
            view_and_capture(cfg.MODEL.NAME, sample_idx, frame_id, 1,
                             "experiments/viz/viewpoints/camera2_sideview2.json")

            stop = True if need_stop else False

            while stop:
                vis.poll_events()
                vis.update_renderer()

            print("clear")
            vis.clear_geometries()

    vis.destroy_window()


if __name__ == '__main__':
    main()
