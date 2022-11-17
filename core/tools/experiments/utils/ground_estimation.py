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


#
# def viz(viz_dict, **kwargs):
#     vis = open3d.visualization.Visualizer()
#     vis.create_window()
#
#     vis.get_render_option().point_size = 1.0
#     vis.get_render_option().background_color = np.ones(3) * 0.5  # 0.8
#     color_map = {'r': np.array([0.7, 0.1, 0.1]),
#                  'g': np.array([0.1, 0.7, 0.1]),
#                  'b': np.array([0.1, 0.1, 0.7]),
#                  'w': np.array([1, 1, 1])
#                  }
#     # draw origin
#     if kwargs.get('draw_origin', False):
#         axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#         vis.add_geometry(axis_pcd)
#
#     def add_cloud(points, color=None):
#         cloud = open3d.geometry.PointCloud()
#         cloud.points = open3d.utility.Vector3dVector(points[:, :3])
#         if color is None:
#             cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
#         else:
#             if len(color.shape) == 1:
#                 color = color[None, :].repeat(points.shape[0], axis=0)
#                 cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))
#             else:
#                 color = color / color.max()
#                 cloud.colors = open3d.utility.Vector3dVector(np.clip(color, 0, 1))
#         vis.add_geometry(cloud)
#         return cloud
#
#     def add_keypoint(points, radius=0.05, color=None):
#         for i in range(points.shape[0]):
#             mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
#             mesh_sphere.translate(points[i])
#             mesh_sphere.compute_vertex_normals()
#             if color is None:
#                 mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
#             else:
#                 mesh_sphere.paint_uniform_color(color)
#             vis.add_geometry(mesh_sphere)
#
#     def add_box(boxes, color=(1, 0, 0)):
#         return V.draw_box(vis, boxes, color)
#
#     count_map = viz_dict['infos'][0]
#     valid_map = viz_dict['infos'][1]
#     height_map = viz_dict['infos'][2]
#     normal_map = viz_dict['infos'][3]
#     min_grid = viz_dict['infos'][4]
#     voxel_size = viz_dict['infos'][5]
#
#     plt.rcParams['savefig.dpi'] = 300  # 图片像素
#     plt.rcParams['figure.dpi'] = 300  # 分辨率 s
#     plt.subplot(1, 3, 1)
#     plt.imshow(count_map, cmap=plt.cm.hot, vmin=0, vmax=count_map.max())
#     plt.colorbar()
#     plt.subplot(1, 3, 2)
#     plt.imshow(valid_map, cmap=plt.cm.hot, vmin=0, vmax=1)
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(height_map, cmap=plt.cm.hot, vmin=height_map.min(), vmax=height_map.max())
#     plt.colorbar()
#
#     plt.show()
#
#     for i, j in zip(*np.where(valid_map > 0)):
#         # mesh_arrow = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05,
#         #                                                        cylinder_height=0.5,
#         #                                                        cone_radius=0.1,
#         #                                                        cone_height=0.25)
#         mesh_arrow = open3d.geometry.TriangleMesh.create_sphere(radius=0.2)
#         mesh_arrow.compute_vertex_normals()
#         normal = normal_map[i, j]
#         height = height_map[i, j]
#         x, y = (i + min_grid[0]) * voxel_size[0], (j + min_grid[1]) * voxel_size[1] - 50
#         trans = np.array((x, y, height), dtype=np.float64)
#         mesh_arrow.paint_uniform_color([0.1, 0.1, 0.7])
#         mesh_arrow.translate(trans)
#         axis = np.cross(normal, np.array((0, 0, 1)))
#         from scipy.spatial.transform import Rotation as R
#         mesh_arrow.rotate(R.from_rotvec(axis).as_matrix())
#         # vis.add_geometry(mesh_arrow)
#
#     cloud = viz_dict['ds_point']
#     # gd_sampling
#     with TimeMeasurement("gds") as t:
#         point_idx = viz_dict['infos'][6]
#         point_valid = 1 - valid_map[(point_idx[:, 0], point_idx[:, 1])]
#         point_count = count_map[(point_idx[:, 0], point_idx[:, 1])]
#         probabilities = (53 - point_count)
#         probabilities = probabilities / probabilities.sum()
#
#         gd_sample_idx = np.random.choice(cloud.shape[0], size=4096, replace=False, p=probabilities)
#         gd_sample_cloud = cloud[gd_sample_idx, :]
#         ignore_idx = np.ones((1, cloud.shape[0]), dtype=bool)
#         ignore_idx[:, gd_sample_idx] = False
#         ignore_idx = np.where(ignore_idx)[1]
#         ignore_cloud = viz_dict['ds_point'][ignore_idx, :]
#
#     # random point sampling
#     with TimeMeasurement("rps") as t:
#         rp_sample_idx = np.random.choice(a=np.arange(0, cloud.shape[0]),
#                                          size=4096, replace=False)
#         rp_sample_cloud = cloud[rp_sample_idx, :]
#
#     # fps sampling
#     with TimeMeasurement("fps") as t:
#         cloud_tensor = torch.from_numpy(cloud).cuda().float()[:, :3].view(1, -1, 3).contiguous()
#         fps_sample_idx = farthest_point_sample(cloud_tensor, 4096).cpu().numpy().reshape(-1)
#         fps_sample_cloud = cloud[fps_sample_idx, :]
#
#     add_cloud(viz_dict['ground'] + np.array([0, 0, -0.01, 0]), color=color_map['r'])
#     add_cloud(viz_dict['ds_point'], color=color_map['w'])  # add_cloud(ignore_cloud, color=color_map['w'])
#     add_cloud(gd_sample_cloud + np.array([0, 40, 0, 0]), color=color_map['w'])
#     add_cloud(fps_sample_cloud + np.array([0, 80, 0, 0]), color=color_map['w'])
#
#     vis.run()
#     vis.destroy_window()


# @numba.jit(nopython=True, cache=True)
# def pca_svd(x):
#     n, c = x.shape
#     mean_x = x[:, 0].mean()
#     mean_y = x[:, 1].mean()
#     mean_z = x[:, 2].mean()
#     mean = np.array([mean_x, mean_y, mean_z])
#     x_centeried = x - mean
#     u, s, v = np.linalg.svd(x_centeried)
#     components = v[:c].T
#     return s, components, mean_z
#
#
# @numba.jit('(float32[:,:],int64[:,:,:], int32[:, :])', nopython=True, cache=True)
# def analysis(points, patchs_point_idx, patchs_point_count):
#     ground_valid_map = np.zeros_like(patchs_point_count, dtype=np.int32)  # ground 1, non_ground 0, unknown -1
#     ground_height_map = np.zeros_like(patchs_point_count, dtype=np.float32)
#     ground_normal = np.zeros((*patchs_point_count.shape, 3), dtype=np.float32)
#
#     # coarse seg
#     for i in range(patchs_point_count.shape[0]):
#         for j in range(patchs_point_count.shape[1]):
#             count = patchs_point_count[i, j]
#             if count > 2:
#                 points_idx = patchs_point_idx[i, j, :count]
#                 patch = points[points_idx, :3]
#                 s, v, mean_z = pca_svd(patch)
#                 if ((np.linalg.norm(np.cross(v[..., 0], [0, 0, 1])) > 0.8
#                      and np.linalg.norm(np.cross(v[..., 1], [0, 0, 1])) > 0.8)
#                         and s[1] / (s[2] + 1e-6) > 5):
#                     ground_height_map[i, j] = mean_z
#                     ground_valid_map[i, j] = 1
#                     ground_normal[i, j] = v[..., 2]
#
#     return ground_normal, ground_height_map, ground_valid_map
#
#
# @numba.jit('(int32[:, :], int32[:, :], int64[:, :, :])', nopython=True, cache=True)
# def generate_patchs_kernel(points_grid_idx, grid_count, point_patchs):
#     max_num_point = point_patchs.shape[2]
#     for i in range(points_grid_idx.shape[0]):
#         grid_x, grid_y = points_grid_idx[i, 0], points_grid_idx[i, 1]
#         cnt = grid_count[grid_x, grid_y]
#         if cnt < max_num_point:
#             point_patchs[grid_x, grid_y, cnt] = i
#             grid_count[grid_x, grid_y] += 1
#
#
# def generate_patch(points, voxel_size):
#     point_grid_idx = np.round(points[:, 0:2] / np.array(voxel_size)).astype(np.int32)
#
#     min_grid_idx = np.min(point_grid_idx, axis=0)
#     max_grid_idx = np.max(point_grid_idx, axis=0)
#     point_grid_idx = point_grid_idx - min_grid_idx
#
#     grid_size = max_grid_idx - min_grid_idx + 1
#     grid_count = np.zeros(grid_size, dtype=np.int32)
#     point_idx_patchs = np.zeros((*grid_size, 50), dtype=np.int64)
#
#     generate_patchs_kernel(point_grid_idx, grid_count, point_idx_patchs)
#
#     return point_grid_idx, point_idx_patchs, grid_count, min_grid_idx, grid_size
#
#
# def ground_estimation(points, voxel_size=[2, 2]):
#     points = points.copy().astype(np.float32)
#     with TimeMeasurement("grid") as t:
#         point_idx, patchs_point_idx, patch_point_counts, min_grid, grid_size = generate_patch(points,
#                                                                                               voxel_size=voxel_size)
#     with TimeMeasurement("ground") as t:
#         normal_dense, height_dense, valid_dense = analysis(points, patchs_point_idx, patch_point_counts)
#     with TimeMeasurement("nonground") as t:
#         non_ground = np.arange(point_idx.shape[0])[valid_dense[point_idx[:, 0], point_idx[:, 1]] == 0]
#         ground = np.ones((1, points.shape[0]), dtype=bool)
#         ground[:, non_ground] = False
#         ground = np.where(ground)[1]
#
#     print(f"grid_size: {grid_size}")
#     print(f"no_empty: {np.array(np.where(patch_point_counts > 0), dtype=np.int32).shape[1]}")
#     print(f"ground: {ground.shape[0]}")
#     print(f"non_ground: {non_ground.shape[0]}")
#     return points[ground, :], \
#            (patch_point_counts, valid_dense, height_dense, normal_dense, min_grid, voxel_size, point_idx)


POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 30]
VOXEL_SIZE = [2, 2, POINT_CLOUD_RANGE[-1] - POINT_CLOUD_RANGE[-1 - 3]]
point_cloud_range = torch.Tensor(POINT_CLOUD_RANGE).cuda().float()
voxel_size = torch.Tensor(VOXEL_SIZE).cuda().float()
grid_size = torch.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).cuda().long()
print(voxel_size, grid_size)
scale_xy = grid_size[0] * grid_size[1]
scale_y = grid_size[1]

num_sample = 16384


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
    with TimeMeasurement("gdsp") as t:
        count = viz_dict['points_count'][0]
        unq_count = viz_dict['unq_count']
        unq_count = unq_count[unq_count < scale_xy.cpu().numpy()]
        rho = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
        phi = np.arctan2(cloud[:, 1], cloud[:, 0])
        print(f"max(count):{np.max(count)}")
        print(f"mean(count):{np.mean(count)}")
        print(f"std(count):{np.std(count)}")
        print(f"max(unq_count):{np.max(unq_count)}")
        print(f"mean(unq_count):{np.mean(unq_count)}")
        print(f"std(unq_count):{np.std(unq_count)}")
        print(f"mean(rho):{np.mean(rho)}")
        print(f"sample_ratio:{cloud.shape[0] / num_sample}")

        # probabilities = np.clip(count, a_max=np.mean(unq_count), a_min=0)
        # 目的：远处使用密度采样(可以以均值为界限)，近处使用距离采样和角度采样。
        with TimeMeasurement("gdsp_sample") as t:
            # # mothod 1
            # dense_item = (1 / count)  # 越远密度越小 概率越大(0,1)
            # probabilities = dense_item
            # probabilities_norm = probabilities / probabilities.sum()
            #
            # gd_sample_idx = np.random.choice(cloud.shape[0], size=num_sample, replace=False, p=probabilities_norm)
            # gd_sample_cloud = cloud[gd_sample_idx, :]

            # mothod 2
            ratio = cloud.shape[0] / num_sample
            unq_phi, phi_inv, unq_count = np.unique(phi // np.deg2rad(0.05), return_inverse=True, return_counts=True)
            phi_wight = np.ones([int(np.floor(ratio))]) / ratio
            phi_wight[0] = 5
            phi_wight = phi_wight[None, ...].repeat(np.ceil(unq_phi.shape[0] / phi_wight.shape[0]), 0).reshape(-1)
            phi_wight = phi_wight[:unq_phi.shape[0]]

            probabilities =  rho
            probabilities_norm = probabilities / probabilities.sum()
            gd_sample_idx = np.random.choice(cloud.shape[0], size=num_sample, replace=False, p=probabilities_norm)
            gd_sample_cloud = cloud[gd_sample_idx, :]

            # method 3

        # ignore_idx = np.ones((1, cloud.shape[0]), dtype=bool)
        # ignore_idx[:, gd_sample_idx] = False
        # ignore_idx = np.where(ignore_idx)[1]
        # ignore_cloud = viz_dict['ds_point'][ignore_idx, :]
    with TimeMeasurement("test") as t:
        np.random.choice(unq_count.shape[0], size=num_sample, replace=True, p=unq_count / unq_count.sum())

    # random point sampling
    with TimeMeasurement("rps") as t:
        rp_sample_idx = np.random.choice(a=np.arange(0, cloud.shape[0]),
                                         size=num_sample, replace=False)
        rp_sample_cloud = cloud[rp_sample_idx, :]

    # fps sampling
    with TimeMeasurement("fps") as t:
        cloud_tensor = torch.from_numpy(cloud).cuda().float()[:, :3].view(1, -1, 3).contiguous()
        fps_sample_idx = farthest_point_sample(cloud_tensor, num_sample).cpu().numpy().reshape(-1)
        fps_sample_cloud = cloud[fps_sample_idx, :]

    # add_cloud(viz_dict['ground'] + np.array([0, 0, -0.01, 0]), color=color_map['r'])
    # add_cloud(viz_dict['ds_point'],
    #           color=probabilities[..., None].repeat(3, -1))  # add_cloud(ignore_cloud, color=color_map['w'])
    add_cloud(viz_dict['ds_point'], color=color_map['w'])
    add_cloud(gd_sample_cloud + np.array([0, 40, 0, 0]), color=color_map['w'])
    add_cloud(fps_sample_cloud + np.array([0, 80, 0, 0]), color=color_map['w'])

    vis.run()
    vis.destroy_window()


from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import dense_aware_point_sampling


def ground_estimation(points):
    points = points[None, ...].repeat(8, axis=0)
    bmask = np.arange(0, 8)[None, ...].repeat(points.shape[1])
    points1 = np.hstack((bmask.reshape(-1, 1), points.reshape(-1, 4)))
    points_gpu = torch.from_numpy(points1).cuda().float()
    ponits_batch_gpu = torch.from_numpy(points).cuda().float()
    with TimeMeasurement('dsp') as t:
        dense_aware_point_sampling(ponits_batch_gpu.contiguous(), num_sample)

    with TimeMeasurement('voxelization') as t:
        pillar_coor = torch.floor((points_gpu[:, [1, 2]] - point_cloud_range[[0, 1]]) / voxel_size[None, [0, 1]]).long()
        voxel_coor_id = points_gpu[:, 0].int() * scale_xy + pillar_coor[:, 0] * scale_y + pillar_coor[:, 1]
        unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
        count_sparse = unq_count[unq_inv].view(8, -1)

    coor_sparse = unq_coor[unq_inv].view(8, -1)
    count_dense = torch.zeros((grid_size[0], grid_size[1])).cuda().long()
    coor = coor_sparse[0]
    grid_idx = torch.stack(((coor % scale_xy) // scale_y, coor % scale_y))
    count_dense[(grid_idx[0], grid_idx[1])] = count_sparse[0]
    count_dense = count_dense.cpu().numpy()
    plt.imshow(count_dense, cmap=plt.cm.hot, vmin=0, vmax=count_dense.max())
    plt.colorbar()
    plt.show()
    return count_sparse, unq_count


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

            points = data_dict['points_raw']
            point_cloud_range_cpu = point_cloud_range.cpu().numpy()
            mask = (points[:, 0] >= point_cloud_range_cpu[0]) & (points[:, 0] <= point_cloud_range_cpu[3]) \
                   & (points[:, 1] >= point_cloud_range_cpu[1]) & (points[:, 1] <= point_cloud_range_cpu[4])
            points = points[mask]
            points = points
            ground_estimation(points)
            infos = ground_estimation(points)
            viz_dict = {'points': points,
                        'ds_point': data_dict['points'],
                        'points_count': infos[0],
                        'unq_count': infos[1]}
            for k, v in viz_dict.items():
                if isinstance(v, torch.Tensor):
                    viz_dict[k] = v.cpu().numpy()

            viz(viz_dict, draw_origin=False)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

            logger.info('Demo done.')


if __name__ == '__main__':
    main()

# @numba.jit(nopython=True, cache=True)
# def pca_svd(x):
#     n, c = x.shape
#     mean_x = x[:, 0].mean()
#     mean_y = x[:, 1].mean()
#     mean_z = x[:, 2].mean()
#     mean = np.array([mean_x, mean_y, mean_z])
#     x_centeried = x - mean
#     u, s, v = np.linalg.svd(x_centeried)
#     components = v[:c].T
#     return s, components, mean_z
#
#
# @numba.jit('(float32[:,:],int64[:,:,:], int32[:, :], int32[:, :])', nopython=True, cache=True)
# def analysis(points, patchs_point_idx, patchs_point_count, grid):
#     ground_points_idx = []
#     ground_normal = []
#     ground_grid_idx = []
#     ground_mean_height = np.ones_like(patchs_point_count, dtype=np.float32) * -1e6
#     # coarse seg
#     for i in range(grid.shape[0]):  # for each no empty patch
#         grid_x, grid_y = grid[i, 0], grid[i, 1]
#         count = patchs_point_count[grid_x, grid_y]
#         if count >= 3:  # three point to define a plane
#             points_idx = patchs_point_idx[grid_x, grid_y, :count]
#             patch = points[points_idx, :3]
#             s, v, mean_z = pca_svd(patch)
#             if (np.linalg.norm(np.cross(v[..., 0], [0, 0, 1])) > 0.8
#                     and np.linalg.norm(np.cross(v[..., 1], [0, 0, 1])) > 0.8
#                     and s[1] / (s[2] + 1e-6) > 5):
#                 ground_points_idx.append(points_idx)
#                 ground_mean_height[grid_x, grid_y] = mean_z
#                 ground_normal.append(v[..., 2])
#                 ground_grid_idx.append([grid_x, grid_y])
#         # else:
#         #     ground_mean_height[grid_x, grid_y] = 0
#
#     return ground_points_idx, ground_normal, ground_grid_idx, ground_mean_height
#
#
# @numba.jit('(float32[:,:], int32[:, :],int32[:],int32[:, :],int64[:, :, :])', nopython=True, cache=True)
# def generate_patchs_kernel(points, points_grid_idx, min_grid_idx, grid_count, point_patchs):
#     min_x, min_y = min_grid_idx[0], min_grid_idx[1]
#     max_num_point = point_patchs.shape[2]
#     for i in range(points.shape[0]):
#         grid_x, grid_y = points_grid_idx[i, 0] - min_x, points_grid_idx[i, 1] - min_y
#         cnt = grid_count[grid_x, grid_y]
#         if cnt < max_num_point:
#             point_patchs[grid_x, grid_y, cnt] = i
#             grid_count[grid_x, grid_y] += 1
#
#
# def generate_patch(points, voxel_size):
#     point_grid_idx = np.round(points[:, 0:2] / np.array(voxel_size)).astype(np.int32)
#
#     min_grid_idx, max_grid_idx = np.min(point_grid_idx, axis=0), np.max(point_grid_idx, axis=0)
#     grid_size = max_grid_idx - min_grid_idx + 1
#     grid_count = np.zeros(grid_size, dtype=np.int32)
#     point_grid_idx_patchs = np.zeros((*grid_size, 50), dtype=np.int64)
#
#     generate_patchs_kernel(points, point_grid_idx, min_grid_idx, grid_count, point_grid_idx_patchs)
#
#     print(f"grid_size:{grid_size}")
#     return point_grid_idx, point_grid_idx_patchs, grid_count, min_grid_idx
#
#
# def ground_estimation(points, grid_size=[1, 1]):
#     points = points.astype(np.float32)
#     with TimeMeasurement("grid") as t:
#         point_idx, patchs_point_idx, patch_point_counts, min_grid = generate_patch(points, voxel_size=grid_size)
#         no_empty_patch = np.array(np.where(patch_point_counts > 0), dtype=np.int32).T
#         print(f"uniqe_idx:{no_empty_patch.shape}")
#     with TimeMeasurement("ground") as t:
#         ground, *normal = analysis(points, patchs_point_idx, patch_point_counts, no_empty_patch)
#         ground = np.concatenate(ground, axis=0)
#     with TimeMeasurement("nonground") as t:
#         mean = normal[2]
#         point_idx = point_idx - min_grid
#         non_ground = np.arange(point_idx.shape[0])[(mean[point_idx[:, 0], point_idx[:, 1]] < -1e5)]
#
#         ground = np.ones((1, points.shape[0]), dtype=np.bool)
#         ground[:, non_ground] = False
#         ground = np.where(ground)[1]
#     print(f"ground: {ground.shape[0]}")
#     print(f"non_ground: {non_ground.shape[0]}")
#     return points[ground, :], (*normal, min_grid, grid_size)
