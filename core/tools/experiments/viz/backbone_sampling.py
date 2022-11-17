import argparse
import glob
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
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
from pcdet.utils import common_utils


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

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        points_raw = input_dict['points'].copy()
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['points_raw'] = points_raw
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def viz(viz_dict, **kwargs):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.ones(3) * 0.8
    color_map = {'r': (0.7, 0.1, 0.1),
                 'g': (0.1, 0.7, 0.1),
                 'b': (0.1, 0.1, 0.7),
                 }
    if kwargs.get('draw_origin', False):
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    def add_cloud(points, color=None):
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points[:, :3])
        if color is None:
            cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
        else:
            cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))
        vis.add_geometry(cloud)

    def add_keypoint(points, radius=0.05, color=None):
        for i in range(points.shape[0]):
            mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh_sphere.translate(points[i])
            mesh_sphere.compute_vertex_normals()
            if color is None:
                mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
            else:
                mesh_sphere.paint_uniform_color(color)
            vis.add_geometry(mesh_sphere)

    def add_box(boxes, color=(1, 0, 0)):
        return V.draw_box(vis, boxes, color)

    print(f"====== marker ======\n"
          f"tiny black points(16384): the points fed to networks\n"
          f"small blue balls(4096): output of downsampling layer 1\n"
          f"middle green balls(1024): output of downsampling layer 2\n"
          f"big red balls(256): candidate points\n"
          f"large golden balls(256): predicted coarse centers\n"
          f"red boxes: predicted box for each coarse center\n"
          f"green boxes: groundtruth boxes")
    add_cloud(viz_dict['points_16384'])
    add_keypoint(viz_dict['points_4096'], radius=0.04, color=color_map['b'])
    add_keypoint(viz_dict['points_1024'], radius=0.06, color=color_map['g'])
    add_keypoint(viz_dict['points_candidate'], radius=0.10, color=color_map['r'])
    add_keypoint(viz_dict['points_vote'], radius=0.15, color=[0.7, 0.7, 0.1])
    vis = add_box(viz_dict['point_box_preds'], color_map['r'])
    vis = add_box(viz_dict['pred_box'], color_map['g'])
    vis.run()
    vis.destroy_window()


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            viz_dict = {'points': data_dict['points_raw']}
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            viz_dict.update({'points_16384': data_dict['points'][:, 1:],
                             'points_4096': data_dict['point_coords_list'][0][:, 1:],
                             'points_1024': data_dict['point_coords_list'][1][:, 1:],
                             'points_512': data_dict['point_coords_list'][2][:, 1:],
                             'points_candidate': data_dict['point_candidate_coords'][:, 1:],
                             'points_vote': data_dict['point_vote_coords'][:, 1:],
                             'point_cls_scores': data_dict['point_cls_scores'],
                             'point_box_preds': data_dict['point_box_preds'],
                             'pred_label': pred_dicts[0]['pred_labels'],
                             'pred_box': pred_dicts[0]['pred_boxes'],
                             'pred_scores': pred_dicts[0]['pred_scores'],
                             })
            for k, v in viz_dict.items():
                if isinstance(v, torch.Tensor):
                    viz_dict[k] = v.cpu().numpy()
            viz(viz_dict, draw_origin=False)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            logger.info('Demo done.')


if __name__ == '__main__':
    main()
