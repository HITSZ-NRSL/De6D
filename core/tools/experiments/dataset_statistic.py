import argparse
import glob
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numba

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
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils.box_utils import points_in_boxes3d


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


def gether_data(dataset):
    box_of_points = []
    for idx, data in enumerate(dataset):  # frame_id/gt_boxes/points
        pts = data['points']
        boxes = data['gt_boxes']
        pts_in_box_indices = points_in_boxes3d(points=pts, boxes3d=boxes)
        max_box_idx, min_box_idx = max(pts_in_box_indices), min(pts_in_box_indices)
        num_pts_in_boxes = np.zeros(boxes.shape[0])
        scene_idx_for_boxes = np.zeros(boxes.shape[0])
        scene_idx_for_boxes.fill(idx)
        for i in range(min_box_idx, max_box_idx + 1):
            num_pts_in_boxes[i] = (pts_in_box_indices == i).sum()
        print(num_pts_in_boxes)

        box_of_points += [np.concatenate((boxes, num_pts_in_boxes[:, None], scene_idx_for_boxes[:, None]), axis=-1)]
        print('collected {}'.format(data['frame_id']))
    return box_of_points


def analysis_scene_cls_pts_num(cls: np.ndarray, num: np.ndarray, scene: np.ndarray, grid_num=100):
    # cls(num_boxes)[cls_id:1~3]: box_cls
    # num(pts_in_boxes)[num_pts]: box_pts_num
    scene = scene.reshape(-1)
    statistic = np.zeros([grid_num + 1, 3])  # (grid_num, cls, val)
    num_one_hot = np.zeros([len(num), int(max(cls)) + 1 - int(min(cls))])
    for c in range(0, num_one_hot.shape[1]):
        num_one_hot[:, c] = (num * (cls == c + 1)).reshape(-1)
    scene_one_hot = np.zeros([int(max(scene)) + 1 - int(min(scene)), int(max(cls)) + 1 - int(min(cls))])
    for s in range(0, scene_one_hot.shape[0]):
        scene_flag = scene == s
        num_one_hot_scene = num_one_hot[scene_flag, :]
        scene_one_hot[s] = num_one_hot_scene.sum(axis=0)

    # 每个类别的box中含有的点的数量最大最小值
    max_, min_ = np.max(scene_one_hot, axis=0), np.min(scene_one_hot, axis=0)
    print('analysis_scene_cls_pts_num: ')
    print('max: {}, min: {}'.format(max_, min_))
    unit_size = (max_ - min_) / grid_num
    indices = np.floor((scene_one_hot - min_) / unit_size).astype(np.int32)
    for row, idx in enumerate(indices):
        statistic[idx[0], 0] += 1
        statistic[idx[1], 1] += 1
        statistic[idx[2], 2] += 1
    range_ = np.concatenate((min_[:, None], max_[:, None]), axis=-1)
    return statistic, range_, grid_num + 1


def analysis_scene_cls_box_num(cls: np.ndarray, num: np.ndarray, grid_num=100):
    statistic = np.zeros([grid_num + 1, 3])
    num_one_hot = np.zeros([len(num), int(max(cls)) + 1 - int(min(cls))])
    num_one_hot.fill(-1)
    for c in range(0, num_one_hot.shape[1]):
        num_one_hot[:, c] = (num * (cls == c + 1)).reshape(-1)

    max_, min_ = np.max(num_one_hot, axis=0), np.min(num_one_hot, axis=0)
    print('analysis_scene_cls_pts_num: ')
    print('max: {}, min: {}'.format(max_, min_))
    unit_size = (max_ - min_) / grid_num
    indices = np.floor((num_one_hot - min_) / unit_size).astype(np.int32)
    for row, idx in enumerate(indices):
        statistic[idx[0], 0] += 1
        statistic[idx[1], 1] += 1
        statistic[idx[2], 2] += 1
    range_ = np.concatenate((min_[:, None], max_[:, None]), axis=-1)
    return statistic, range_, grid_num + 1


def analysis_box_residual(lwh, cls: np.ndarray, num, grid_num=100):
    mean_size = np.array([
        [3.9, 1.6, 1.56],
        [0.8, 0.6, 1.73],
        [1.76, 0.6, 1.73]
    ])
    statistic = np.zeros([grid_num + 1, 3])
    residual = lwh - mean_size[cls.astype(np.int64) - 1].reshape(-1, 3)
    max_, min_ = np.max(residual, axis=0), np.min(residual, axis=0)
    print('analysis box residual: ')
    print('max: {}, min: {}'.format(max_, min_))
    unit_size = (max_ - min_) / grid_num
    indices = np.floor((residual - min_) / unit_size).astype(np.int32)
    for raw, idx in enumerate(indices):
        statistic[idx[0], 0] += num[raw]
        statistic[idx[1], 1] += num[raw]
        statistic[idx[2], 2] += num[raw]
    range_ = np.concatenate((min_[:, None], max_[:, None]), axis=-1)
    return statistic, range_, grid_num + 1


def analysis_box_rotation(rot, num, grid_num=100, for_point=True):
    statistic = np.ones([grid_num + 1, 3])
    max_, min_ = np.max(rot, axis=0), np.min(rot, axis=0)
    print('analysis box rotation: ')
    print('max: {}, min: {}'.format(max_, min_))
    unit_size = (max_ - min_) / grid_num
    indices = np.floor((rot - min_) / unit_size).astype(np.int32)
    if for_point:
        for raw, idx in enumerate(indices):
            statistic[idx[0], 0] += num[raw]
            statistic[idx[1], 1] += num[raw]
            statistic[idx[2], 2] += num[raw]
    else:  # for box
        for raw, idx in enumerate(indices):
            statistic[idx[0], 0] += 1
            statistic[idx[1], 1] += 1
            statistic[idx[2], 2] += 1
    range_ = np.concatenate((min_[:, None], max_[:, None]), axis=-1)

    return statistic, range_, grid_num + 1


def main():
    args, cfg = parse_config()
    file_name = 'experiments/results/dataset_statistic.npy'
    if not pathlib.Path(file_name).exists():
        aug_dataset = KittiDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path=None,
            training=True
        )
        box_of_points = gether_data(aug_dataset)
        box_of_points = np.concatenate(box_of_points, axis=0)
        print('box_of_poitns: shape:{}, type:{}'.format(box_of_points.shape, box_of_points.dtype))
        with open(file_name, 'w') as f:
            box_of_points.tofile(f)
    else:
        box_of_points = np.fromfile(file_name, dtype=np.float64).reshape(-1, 12)

    # [xyz, lwh, rzyx, cls, num, idx, _] = np.split(box_of_points, [3, 6, 9, 10, 11, 12], axis=1)
    #
    # # analysis
    # pts_res, range_pts_res, grid_num_pts_res = analysis_box_residual(lwh, cls, num, grid_num=100)
    # pts_rot, range_pts_rot, grid_num_pts_rot = analysis_box_rotation(rzyx, num, grid_num=100)
    # box_cls, range_box_cls, grid_num_box_cls = analysis_scene_cls_box_num(cls, num, grid_num=100)
    # box_rot, range_box_rot, grid_num_box_rot = analysis_box_rotation(rzyx, num, grid_num=100, for_point=False)
    # pts_cls, range_pts_cls, grid_num_pts_cls = analysis_scene_cls_pts_num(cls, num, idx, grid_num=100)
    #
    # # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(6.4 * 4, 4.8 * 2))
    #
    # # plot: distribution of xy
    # plt.subplot(2, 3, 1)
    # plt.title("distribution of object position, sampled each 100 one")
    # plt.xlabel("x/m")
    # plt.ylabel("y/m")
    # plt.scatter(xyz[:, 0][::100], xyz[:, 1][::100], alpha=0.5)
    #
    # # plot: distribution of box rotation
    # plt.subplot(2, 3, 2)
    # plt.title("distribution of box rotation")
    # plt.xlabel("rotation/rad")
    # plt.ylabel("the number of boxes/lg(n)")
    # line = plt.plot(np.linspace(range_box_rot[:, 0], range_box_rot[:, 1], grid_num_box_rot), np.log10(box_rot))
    # plt.legend(handles=line, labels=['yaw', 'pitch', 'roll'], loc='best')
    #
    # # plot: distribution of box cls
    # plt.subplot(2, 3, 3)
    # plt.title("distribution of the number of point in box")
    # plt.xlabel("the number of point in box/n")
    # plt.ylabel("the num of box/lg(n)")
    # line = plt.plot(np.linspace(range_box_cls[:, 0], range_box_cls[:, 1], grid_num_box_cls - 1),
    #                 np.log10(box_cls[1:, :]))
    # plt.legend(handles=line, labels=['Car', 'Pedestrian', 'Cyclist'], loc='best')
    #
    # # plot: distribution of lwh residual
    # plt.subplot(2, 3, 4)
    # plt.title("distrustion of each point ped box dim residual")
    # plt.xlabel("box residual/m")
    # plt.ylabel("the number of points")
    # line = plt.plot(np.linspace(range_pts_res[:, 0], range_pts_res[:, 1], grid_num_pts_res), pts_res)
    # plt.legend(handles=line, labels=['l', 'w', 'h'], loc='best')
    #
    # # plt: rotation distrustion of box associated each pts
    # plt.subplot(2, 3, 5)
    # plt.title("distribution of each point pred box rotation")
    # plt.xlabel("rotation/rad")
    # plt.ylabel("the number of points/lg(n)")
    # line = plt.plot(np.linspace(range_pts_rot[:, 0], range_pts_rot[:, 1], grid_num_pts_rot), np.log10(pts_rot))
    # plt.legend(handles=line, labels=['yaw', 'pitch', 'roll'], loc='best')
    # print(pts_rot[:, 1][:-1].sum(), pts_rot[:, 1][-1])
    #
    # # plt: positve pts number distribution of all scenes
    # plt.subplot(2, 3, 6)
    # plt.title("distribution of each class points in scene")
    # plt.xlabel("the number of points/n")
    # plt.ylabel("the number of scene/n")
    # line = plt.plot(np.linspace(range_pts_cls[:, 0], range_pts_cls[:, 1], grid_num_pts_cls), pts_cls)
    # plt.legend(handles=line, labels=['Car', 'Pedestrian', 'Cyclist'], loc='best')
    #
    # plt.show()


if __name__ == '__main__':
    main()
