import argparse
import glob
import os
import sys
from pathlib import Path
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt

# this_path = os.getcwd()
# os.chdir('../../tools')
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
from pcdet.datasets.slopekitti.kitti_dataset import SlopeKittiDataset


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
    parser.add_argument('--pause', type=bool, default=False)

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

    data_list = range(0, len(demo_dataset))

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

    """
    create window weight
    """
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1575, height=587)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array(color_map['bg'])
    vis.register_key_action_callback(32, key_action_callback)

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
            pred_dicts, _ = model.forward(data_dict)

            print(f"get prediction")
            points = data_dict['raw_points'][0]
            # points = data_dict['points'][:, 1:4]
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_boxes = pred_boxes[pred_dicts[0]['pred_labels'] == 1].cpu().numpy()

            print(f"points: {points.shape}, pred_boxes: {pred_boxes.shape}")

            # add cloud
            points_draw = points.cpu().numpy()
            add_cloud(vis, points_draw, color=color_map['bg_poitns'])

            # add box
            vis = V.draw_box(vis, pred_boxes, color_map['pred_boxes'], width=7)

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

                output_dir = Path('experiments/results/gtav')
                file_name = f"{model_name}_{idx}_{frame_id}_{i}.png"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / file_name
                plt.imsave(output_path.__str__(), img)

            vis.get_render_option().point_size = 3.5
            view_and_capture(cfg.MODEL.NAME, sample_idx, frame_id, 0, "experiments/viz/viewpoints/pv1.json")

            stop = True if need_stop else False
            while stop:
                vis.poll_events()
                vis.update_renderer()

            print("clear")
            vis.clear_geometries()

    vis.destroy_window()


if __name__ == '__main__':
    main()

# else:
#     open3d.visualization.gui.Application.instance.initialize()
#     vis = open3d.visualization.O3DVisualizer()
#     vis.set_background(np.concatenate((color_map['bg'], [1])), None)
#     vis.show_skybox(False)
#     vis.line_width = 3
#     vis.point_size = 2
#
#     if True:  # cloud
#         in_box_mask = box_utils.points_in_boxes3d(points, gt_boxes[:, :9])
#         in_box_mask = in_box_mask >= 0
#         bg_points = points[in_box_mask == 0, :].cpu().numpy()
#         fg_points = points[in_box_mask, :].cpu().numpy()
#
#         add_cloud(vis, bg_points, color=color_map['bg_poitns'], n=0)
#         add_keypoint(vis, fg_points, radius=0.03, color=color_map['fg_poitns'], n=0)
#     if True:  # box
#         vis = V.draw_box(vis, pred_boxes.cpu().numpy(), color_map['pred_boxes'], width=5, n=0)
#         # vis = V.draw_box(vis, gt_boxes.cpu().numpy(), color_map['gt_boxes'], width=5, n=1)
#     open3d.ml.vis.Visualizer()
#     vis.reset_camera_to_default()
#     vis.setup_camera(60.0, [36.397695375651658, -1.6231406978367948, -4.2950364593215342],
#                      [-0.98272805435546617, 0.048799986180770846, 0.17850527311952574],
#                      [0.17839905277756171, -0.0065589050546949737, 0.9839363590865785])
#     vis.animation_time_step = 1.0
#     open3d.visualization.gui.Application.instance.add_window(vis)
#     open3d.visualization.gui.Application.instance.run()
