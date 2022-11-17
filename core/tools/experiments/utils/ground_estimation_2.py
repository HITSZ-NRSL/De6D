import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numba

this_path = os.getcwd()
os.chdir('../../tools')
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
from pcdet.utils import common_utils
from pcdet.utils.common_utils import TimeMeasurement

import matplotlib.pyplot as plt
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import farthest_point_sample


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


POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 30]
VOXEL_SIZE = [2, 2, POINT_CLOUD_RANGE[-1] - POINT_CLOUD_RANGE[-1 - 3]]
point_cloud_range = torch.Tensor(POINT_CLOUD_RANGE).cuda().float()
voxel_size = torch.Tensor(VOXEL_SIZE).cuda().float()
grid_size = torch.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).cuda().long()
print(voxel_size, grid_size)
scale_xy = grid_size[0] * grid_size[1]
scale_y = grid_size[1]

num_sample = 4096


def viz(viz_dict, **kwargs):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3) * 0.5  # 0.8
    color_map = {'r': np.array([0.7, 0.1, 0.1]),
                 'g': np.array([0.1, 0.7, 0.1]),
                 'b': np.array([0.1, 0.1, 0.7]),
                 'w': np.array([1, 1, 1])
                 }
    # draw origin
    if kwargs.get('draw_origin', False):
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    def add_cloud(points, color=None):
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points[:, :3])
        if color is None:
            cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
        else:
            if len(color.shape) == 1:
                color = color[None, :].repeat(points.shape[0], axis=0)
                cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))
            else:
                color = color / color.max()
                color = np.power(color, 0.8)
                cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))
        vis.add_geometry(cloud)
        return cloud

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

    cloud = viz_dict['points']
    # gd_sampling
    dasp_cube_sample_cloud = cloud[viz_dict['dasp_idx'], :]
    dasp_cyld_sample_cloud = cloud[viz_dict['cyld_idx'], :]

    # fps sampling
    with TimeMeasurement("fps") as t:
        cloud_tensor = torch.from_numpy(cloud).cuda().float()[:, :3].view(1, -1, 3).contiguous()
        fps_sample_idx = farthest_point_sample(cloud_tensor, num_sample).cpu().numpy().reshape(-1)
        fps_sample_cloud = cloud[fps_sample_idx, :]

    # random point sampling
    with TimeMeasurement("rps") as t:
        rp_sample_idx = np.random.choice(a=np.arange(0, cloud.shape[0]),
                                         size=num_sample, replace=False)
        rps_sample_cloud = cloud[rp_sample_idx, :]

    # statistic
    def statistic_cloud(pts, size=50, grid=0.1):
        dist_hist = np.zeros([int(size / grid)])
        dist = np.linalg.norm(pts[:, 0:3], axis=-1)
        for i in range(int(size / grid)):
            flag = np.logical_and(i * grid < dist, dist < (i + 1) * grid)
            dist_hist[i] = flag.sum()
        return dist_hist

    rows, cols = 6, 2
    plt.figure(figsize=(6, 10))

    plt.subplot(rows, cols, 1)
    cloud_hist = statistic_cloud(cloud)
    rps_hist = statistic_cloud(rps_sample_cloud)
    plt.title("raw v.s. rps")
    plt.plot(cloud_hist, label='raw')
    plt.plot(rps_hist, label='rps')
    plt.subplot(rows, cols, 2)
    error = (rps_hist - cloud_hist) / (1 + rps_hist)
    plt.plot(error)

    plt.subplot(rows, cols, 3)
    dasp_cube_hist = statistic_cloud(dasp_cube_sample_cloud)
    plt.title("raw v.s. dasp cube")
    plt.plot(cloud_hist, label='raw')
    plt.plot(dasp_cube_hist, label='dasp_cube')
    plt.subplot(rows, cols, 4)
    error = (dasp_cube_hist - cloud_hist) / (1 + dasp_cube_hist)
    plt.plot(error)

    plt.subplot(rows, cols, 5)
    dasp_cyld_hist = statistic_cloud(dasp_cyld_sample_cloud)
    plt.title("raw v.s. dasp cyld")
    plt.plot(cloud_hist, label='raw')
    plt.plot(dasp_cyld_hist, label='dasp_cyld')
    plt.subplot(rows, cols, 6)
    error = (dasp_cyld_hist - cloud_hist) / (1 + dasp_cyld_hist)
    plt.plot(error)

    plt.subplot(rows, cols, 7)
    plt.title("raw v.s. fps")
    fps_hist = statistic_cloud(fps_sample_cloud)
    plt.plot(cloud_hist, label='raw')
    plt.plot(fps_hist, label='fps')
    plt.subplot(rows, cols, 8)
    error = (fps_hist - cloud_hist) / (1 + fps_hist)
    plt.plot(error)

    plt.subplot(rows, cols, 9)
    plt.title("fps v.s. dasp cube")
    plt.plot(fps_hist, label='fps')
    plt.plot(dasp_cube_hist, label='dasp_cube')
    plt.subplot(rows, cols, 10)
    error = (dasp_cube_hist - fps_hist) / (1 + dasp_cube_hist)
    plt.plot(error)
    print(np.std(error))

    plt.subplot(rows, cols, 11)
    plt.title("fps v.s. dasp cyld")
    plt.plot(fps_hist, label='fps')
    plt.plot(dasp_cyld_hist, label='dasp_cyld')
    plt.subplot(rows, cols, 12)
    error = (dasp_cyld_hist - fps_hist) / (1 + dasp_cyld_hist)
    plt.plot(error)
    print(np.std(error))

    plt.tight_layout()
    plt.show()

    add_cloud(viz_dict['ds_point'], color=color_map['w'])
    add_cloud(dasp_cube_sample_cloud + np.array([0, 40, 0, 0]), color=color_map['w'])
    add_cloud(dasp_cyld_sample_cloud + np.array([0, 80, 0, 0]), color=color_map['w'])
    add_cloud(fps_sample_cloud + np.array([0, 120, 0, 0]), color=color_map['w'])
    add_cloud(cloud + np.array([0, 160, 0, 0]), color=color_map['w'])
    vis.run()
    vis.destroy_window()


from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import dense_aware_point_sampling

batch_size = 8


def dense_aware_sampling_cube(points):
    points = points[None, ...].repeat(batch_size, axis=0)
    bmask = np.arange(0, batch_size)[None, ...].repeat(points.shape[1])
    points_gpu = torch.from_numpy(np.hstack((bmask.reshape(-1, 1), points.reshape(-1, 4)))).cuda().float()

    with TimeMeasurement('density_cube') as t:
        pillar_coor = torch.floor((points_gpu[:, [1, 2]] - point_cloud_range[[0, 1]]) / voxel_size[None, [0, 1]]).long()
        voxel_coor_id = points_gpu[:, 0].int() * scale_xy + pillar_coor[:, 0] * scale_y + pillar_coor[:, 1]
        unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
        mean_count = torch.mean(unq_count.float()).item()
        print(f"mean: {mean_count}")
        # unq_count = torch.clip(unq_count, min=0, max=mean_count)
        pointwise_density = unq_count[unq_inv].view(batch_size, -1)
        probabilities = (1 / pointwise_density)  # 越远密度越小 概率越大(0,1)
        for i in range(batch_size):
            probabilities_norm = probabilities[i]
            probabilities_norm /= probabilities_norm.sum()
            probabilities_norm = probabilities_norm.cpu().numpy()
            gd_sample_idx = np.random.choice(int(points_gpu.shape[0] / batch_size),
                                             size=num_sample,
                                             p=probabilities_norm,
                                             replace=False)

    # # viz
    # count_map = torch.zeros((grid_size[0], grid_size[1])).cuda()
    # coor_sparse = unq_coor[unq_inv].view(batch_size, -1)
    # coor = coor_sparse[0]
    # grid_idx = torch.stack(((coor % scale_xy) // scale_y, coor % scale_y))
    # count_map[(-1 - grid_idx[0], -1 - grid_idx[1])] = torch.log10(pointwise_density[0])
    #
    # count_map = count_map.cpu().numpy()
    # plt.imshow(count_map, cmap=plt.cm.hot, vmin=0, vmax=count_map.max())
    # plt.colorbar()
    # plt.show()
    return gd_sample_idx


def dense_aware_sampling_cylinder(points):
    points = points[None, ...].repeat(batch_size, axis=0)
    bmask = np.arange(0, batch_size)[None, ...].repeat(points.shape[1])
    points_gpu = torch.from_numpy(np.hstack((bmask.reshape(-1, 1), points.reshape(-1, 4)))).cuda().float()

    cylinder_voxel_size = points_gpu.new_tensor([0.1, np.pi * 2])  # 1m,5°
    cylinder_range = points_gpu.new_tensor([0, 0,
                                            torch.norm(point_cloud_range[3:5] - point_cloud_range[0:2]),
                                            2 * np.pi])
    cylinder_grid_size = torch.round((cylinder_range[2:4] - cylinder_range[0:2]) / cylinder_voxel_size).cuda().long()
    scale_rho_phi = cylinder_grid_size[0] * cylinder_grid_size[1]
    scale_phi_ = cylinder_grid_size[1]

    with TimeMeasurement('density_cylinder') as t:
        rho = torch.norm(points_gpu[:, 1:3], dim=-1)  # xoy
        phi = torch.atan2(points_gpu[:, 2], points_gpu[:, 1]) + np.pi / 2
        cylinder_coords = torch.hstack((rho[..., None], phi[..., None]))
        pillar_coor = torch.floor(cylinder_coords / cylinder_voxel_size).long()
        voxel_coor_id = points_gpu[:, 0].int() * scale_rho_phi + pillar_coor[:, 0] * scale_phi_ + pillar_coor[:, 1]
        unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
        std_count, mean_count = torch.std_mean(unq_count.float())
        print(f"mean: {mean_count}, std: {std_count}")
        # unq_count = torch.clip(unq_count, min=0, max=mean_count + 1.5 * std_count)
        pointwise_density = unq_count[unq_inv].view(batch_size, -1)
        probabilities = (1 / pointwise_density)  # 越远密度越小 概率越大(0,1)
        for i in range(batch_size):
            probabilities_norm = probabilities[i]
            probabilities_norm /= probabilities_norm.sum()
            probabilities_norm = probabilities_norm.cpu().numpy()
            gd_sample_idx = np.random.choice(int(points_gpu.shape[0] / batch_size),
                                             size=num_sample,
                                             p=probabilities_norm,
                                             replace=False)

    # # viz
    # count_map = torch.zeros((cylinder_grid_size[0], cylinder_grid_size[1])).cuda()
    # coor_sparse = unq_coor[unq_inv].view(batch_size, -1)
    # coor = coor_sparse[0]
    # grid_idx = torch.stack(((coor % scale_rho_phi) // scale_phi_, coor % scale_phi_))
    # count_map[(-1 - grid_idx[0], -1 - grid_idx[1])] = torch.log10(pointwise_density[0])
    #
    # count_map = count_map.cpu().numpy()
    # plt.imshow(count_map, cmap=plt.cm.hot, vmin=0, vmax=count_map.max())
    # plt.colorbar()
    # plt.show()
    return gd_sample_idx


from numba import cuda


@cuda.jit()
def aaa(x):
    print(cuda.gridDim.x, cuda.threadIdx.x)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    with torch.no_grad():
        os.chdir(this_path)
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            # x = torch.zeros([8, 4]).cuda()
            # aaa[8, 1](x)
            points = data_dict['points_raw']
            point_cloud_range_cpu = point_cloud_range.cpu().numpy()
            mask = (points[:, 0] >= point_cloud_range_cpu[0]) & (points[:, 0] <= point_cloud_range_cpu[3]) \
                   & (points[:, 1] >= point_cloud_range_cpu[1]) & (points[:, 1] <= point_cloud_range_cpu[4])
            points = points[mask]
            points = points
            dasp_idx = dense_aware_sampling_cube(points)
            cyld_idx = dense_aware_sampling_cylinder(points)
            viz_dict = {'points': points,
                        'ds_point': data_dict['points'],
                        'dasp_idx': dasp_idx,
                        'cyld_idx': cyld_idx}
            for k, v in viz_dict.items():
                if isinstance(v, torch.Tensor):
                    viz_dict[k] = v.cpu().numpy()

            viz(viz_dict, draw_origin=False)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            logger.info('Demo done.')


if __name__ == '__main__':
    main()
