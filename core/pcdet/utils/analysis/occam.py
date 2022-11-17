from abc import ABC

import numpy as np
import torch
import copy
import easydict
import torch
import open3d
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from spconv.pytorch.utils import PointToVoxel
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class BaseDataset(DatasetTemplate):
    """
    OpenPCDet dataset to load and preprocess the point cloud
    """

    def __init__(self, data_config, class_names, occam_config):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
             occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
        """
        super().__init__(dataset_cfg=data_config, class_names=class_names,
                         training=False)
        self.occam_config = occam_config

    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet cfg

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """

        if source_file_path.split('.')[-1] == 'bin':
            points = np.fromfile(source_file_path, dtype=np.float32)
            points = points.reshape(-1, 4)
        elif source_file_path.split('.')[-1] == 'npy':
            points = np.load(source_file_path)
        else:
            raise NotImplementedError

        # FOV crop is usually done using the image
        if self.occam_config.FOV_CROP:
            angles = np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0])))
            mask = angles <= self.occam_config.FOV_ANGLE
            points = points[mask, :]

        input_dict = {
            'points': points
        }

        # data_dict = self.prepare_data(data_dict=input_dict)
        # pcl = data_dict['points']
        pcl = input_dict['points']
        return pcl


class OccamInferenceDataset(DatasetTemplate):
    """
    OpenPCDet dataset for occam inference; in each iteration a sub-sampled
    point cloud according occam config is generated
    """

    def __init__(self, data_config, class_names, occam_config, pcl, nr_it, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
                dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
            occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
            pcl : ndarray (N, 4)
                preprocessed full point cloud
            nr_it : int
                number of sub-sampling iterations
            logger : Logger
        """
        super().__init__(
            dataset_cfg=data_config, class_names=class_names, training=False,
            root_path=None, logger=logger
        )

        self.occam_config = occam_config
        self.pcl = pcl
        self.logger = logger
        self.nr_it = nr_it

        self.sampling_rand_rot = self.occam_config.SAMPLING.RANDOM_ROT  # ou: 随机旋转,度
        self.sampling_vx_size = np.array(self.occam_config.SAMPLING.VOXEL_SIZE)  # ou: 采样步骤的体素大小
        self.lbda = self.occam_config.SAMPLING.LAMBDA  # ou: 概率调节因子，使得在25m处的点采样到的概率为0.15
        self.sampling_density_coeff = np.array(self.occam_config.SAMPLING.DENSITY_DISTR_COEFF)  # ou: 密度随着距离变化的多项式拟合结果
        # ou: 在随机旋转的各种情况下点云尺寸的最大最小值
        self.sampling_range = self.get_sampling_range(
            rand_rot=self.sampling_rand_rot,
            pcl=self.pcl,
            vx_size=self.sampling_vx_size
        )
        # ou: 体素化句柄
        self.voxel_generator = PointToVoxel(
            vsize_xyz=list(self.sampling_vx_size),
            coors_range_xyz=list(self.sampling_range),
            num_point_features=3,
            max_num_points_per_voxel=self.occam_config.SAMPLING.MAX_PTS_PER_VOXEL,
            max_num_voxels=self.occam_config.SAMPLING.MAX_VOXELS
        )

    def get_sampling_range(self, rand_rot, pcl, vx_size):
        """
        compute min/max sampling range for given random rotation

        Parameters
        ----------
        rand_rot : float
            max random rotation before sampling (+/-) in degrees
        pcl : ndarray (N, 4)
            full point cloud
        vx_size : ndarray (3)
            voxel size for sampling in x, y, z

        Returns
        -------
        sampling_range : ndarray (6)
            min/max sampling range for given rotation
        """
        rotmat_pos = Rotation.from_rotvec([0, 0, rand_rot], degrees=True)
        rotmat_neg = Rotation.from_rotvec([0, 0, -rand_rot], degrees=True)

        rot_pts = np.concatenate(
            (np.matmul(rotmat_pos.as_matrix(), pcl[:, :3].T),
             np.matmul(rotmat_neg.as_matrix(), pcl[:, :3].T)), axis=1)

        min_grid = np.floor(np.min(rot_pts, axis=1) / vx_size) * vx_size - vx_size
        max_grid = np.ceil(np.max(rot_pts, axis=1) / vx_size) * vx_size + vx_size

        sampling_range = np.concatenate((min_grid, max_grid))
        return sampling_range

    def __len__(self):
        return self.nr_it

    def __getitem__(self, index):
        if index == self.nr_it:
            raise IndexError

        # randomly rotate and translate full pcl
        rand_transl = np.random.rand(1, 3) * (self.sampling_vx_size[None, :])
        rand_transl -= self.sampling_vx_size[None, :] / 2
        # ou: [-rand_rot,rand_rot]
        rand_rot_ = np.random.rand(1) * self.sampling_rand_rot * 2 \
                    - self.sampling_rand_rot
        rand_rot_mat = Rotation.from_rotvec([0, 0, rand_rot_[0]], degrees=True)
        rand_rot_mat = rand_rot_mat.as_matrix()
        # ou: rot + trans
        rand_rot_pcl = np.matmul(rand_rot_mat, self.pcl[:, :3].T).T
        rand_rot_transl_pcl = rand_rot_pcl + rand_transl
        rand_rot_transl_pcl = np.ascontiguousarray(rand_rot_transl_pcl)

        pt_keep_mask = np.zeros(self.pcl.shape[0], dtype=np.bool)
        if self.occam_config.SAMPLING.NAME == 'Sampling':
            # voxelixe full pcl
            _, vx_coord, _, pt_vx_id = self.voxel_generator.generate_voxel_with_id(
                torch.from_numpy(rand_rot_transl_pcl))
            vx_coord, pt_vx_id = vx_coord.numpy(), pt_vx_id.numpy()
            vx_coord = vx_coord[:, [2, 1, 0]]  # ou: zyx -> xyz

            # compute voxel center in original pcl
            vx_orig_coord = vx_coord * self.sampling_vx_size[None, :]
            vx_orig_coord += self.sampling_range[:3][None, :]  # ou: min_range
            vx_orig_coord += self.sampling_vx_size[None, :] / 2
            vx_orig_coord -= rand_transl  # ou:复原随机旋转和平移
            vx_orig_coord = np.matmul(np.linalg.inv(rand_rot_mat), vx_orig_coord.T).T

            vx_dist = np.linalg.norm(vx_orig_coord, axis=1)  # ou: 计算距离
            vx_keep_prob = self.lbda * (
                    np.power(vx_dist, 2) * self.sampling_density_coeff[0]
                    + vx_dist * self.sampling_density_coeff[1]
                    + self.sampling_density_coeff[2])

            vx_keep_ids = np.where(np.random.rand(vx_keep_prob.shape[0]) < vx_keep_prob)[0]  # ou: 采样到的体素
            # ou: in 1d，即在第二个参数中找第一个参数，找到删除对应第一列的true。用来获取点是否在保留体素中
            pt_keep_mask = np.in1d(pt_vx_id, vx_keep_ids)

        input_dict = {
            'points': self.pcl[pt_keep_mask, :],
            'mask': pt_keep_mask
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


class OccAM(object):
    """
    OccAM base class to store model, cfg and offer operations to preprocess the
    data and compute the attribution maps
    """

    def __init__(self, data_config, model_config, occam_config, class_names,
                 model_ckpt_path, nr_it, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            model_config : EasyDict
               object detection model definition (OpenPCDet)
            occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
            class_names :
                list of class names (OpenPCDet)
            model_ckpt_path: str
                path to pretrained model weights
            nr_it : int
                number of sub-sampling iterations; the higher, the more accurate
                are the resulting attribution maps
            logger: Logger
        """
        self.data_config = data_config
        self.model_config = model_config
        self.occam_config = occam_config
        self.class_names = class_names
        self.logger = logger
        self.nr_it = nr_it

        self.base_dataset = BaseDataset(data_config=self.data_config,
                                        class_names=self.class_names,
                                        occam_config=self.occam_config)

        self.model = build_network(model_cfg=self.model_config,
                                   num_class=len(self.class_names),
                                   dataset=self.base_dataset)
        self.model.load_params_from_file(filename=model_ckpt_path,
                                         logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    # ou: 读取输入数据并进行occam的预处理。
    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet
        data config using the base dataset

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """
        pcl = self.base_dataset.load_and_preprocess_pcl(source_file_path)
        return pcl

    # ou: 获取原始点云输入的预测
    def get_base_predictions(self, pcl):
        """
        get all K detections in full point cloud for which attribution maps will
        be determined

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)

        Returns
        -------
        base_det_boxes : ndarray (K, 7)
            bounding box parameters of detected objects
        base_det_labels : ndarray (K)
            labels of detected objects
        base_det_scores : ndarray (K)
            confidence scores for detected objects
        """
        input_dict = {
            'points': pcl
        }
        # ou: 又进行依次预处理？
        data_dict = self.base_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.base_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            base_pred_dict, _ = self.model.forward(data_dict)

        base_det_boxes = base_pred_dict[0]['pred_boxes'].cpu().numpy()
        base_det_labels = base_pred_dict[0]['pred_labels'].cpu().numpy()
        base_det_scores = base_pred_dict[0]['pred_scores'].cpu().numpy()

        return base_det_boxes, base_det_labels, base_det_scores

    # ou: 因为batch内每帧点云检测个数不一样因此不能用(B,n,7)来表达，而是用(L,7)来表达，并提供一个对应的batch_id向量
    def merge_detections_in_batch(self, det_dicts):
        """
        In order to efficiently determine the confidence score for
        all detections in a batch they are merged.

        Parameters
        ----------
        det_dicts : list
            list of M dicts containing the detections in the M samples within
            the batch (pred boxes, pred scores, pred labels)

        Returns
        -------
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the M samples
        pert_det_labels : ndarray (L)
            labels of all L detections in the M samples
        pert_det_scores : ndarray (L)
            scores of all L detections in the M samples
        batch_ids : ndarray (L)
            Mapping of the detections to the individual samples within the batch
        """
        batch_ids = []

        data_dict = defaultdict(list)
        # ou:
        for batch_id, cur_sample in enumerate(det_dicts):
            batch_ids.append(
                np.ones(cur_sample['pred_labels'].shape[0], dtype=int)
                * batch_id)

            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_ids = np.concatenate(batch_ids, axis=0)

        merged_dict = {}
        for key, val in data_dict.items():
            if key in ['pred_boxes', 'pred_scores', 'pred_labels']:
                merged_data = []
                for data in val:
                    data = data.cpu().numpy()
                    merged_data.append(data)
                merged_dict[key] = np.concatenate(merged_data, axis=0)

        pert_det_boxes = merged_dict['pred_boxes']
        pert_det_labels = merged_dict['pred_labels']
        pert_det_scores = merged_dict['pred_scores']
        return pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids

    def compute_iou(self, base_boxes, pert_boxes):
        """
        3D IoU between base and perturbed detections
        """
        base_boxes = torch.from_numpy(base_boxes)
        pert_boxes = torch.from_numpy(pert_boxes)
        base_boxes, pert_boxes = base_boxes.cuda(), pert_boxes.cuda()

        iou = boxes_iou3d_gpu(base_boxes[:, :7], pert_boxes[:, :7])
        iou = iou.cpu().numpy()
        return iou

    def compute_translation_score(self, base_boxes, pert_boxes):
        """
        translation score (see paper for details)
        """
        translation_error = np.linalg.norm(
            base_boxes[:, :3][:, None, :] - pert_boxes[:, :3], axis=2)
        translation_score = 1 - translation_error
        translation_score[translation_score < 0] = 0
        return translation_score

    def compute_orientation_score(self, base_boxes, pert_boxes, ind: int):
        """
        orientation score (see paper for details)
        """
        boxes_a = copy.deepcopy(base_boxes)
        boxes_b = copy.deepcopy(pert_boxes)

        # ou: yaw -> [-pi,pi]
        boxes_a[:, ind] = boxes_a[:, ind] % (2 * math.pi)
        boxes_a[boxes_a[:, ind] > math.pi, ind] -= 2 * math.pi
        boxes_a[boxes_a[:, ind] < -math.pi, ind] += 2 * math.pi
        boxes_b[:, ind] = boxes_b[:, ind] % (2 * math.pi)
        boxes_b[boxes_b[:, ind] > math.pi, ind] -= 2 * math.pi
        boxes_b[boxes_b[:, ind] < -math.pi, ind] += 2 * math.pi

        # ou: 获取角度差的最小值，因为一圈有两个方向。
        orientation_error_ = np.abs(
            boxes_a[:, ind][:, None] - boxes_b[:, ind][None, :])
        orientation_error__ = 2 * math.pi - np.abs(
            boxes_a[:, ind][:, None] - boxes_b[:, ind][None, :])
        orientation_error = np.concatenate(
            (orientation_error_[:, :, None], orientation_error__[:, :, None]),
            axis=2)

        orientation_error = np.min(orientation_error, axis=2)
        orientation_score = 1 - (orientation_error if ind == 6 else orientation_error * 4)
        orientation_score[orientation_score < 0] = 0
        return orientation_score

    def compute_scale_score(self, base_boxes, pert_boxes):
        """
        scale score (see paper for details)
        """
        boxes_centered_a = copy.deepcopy(base_boxes)
        boxes_centered_b = copy.deepcopy(pert_boxes)
        boxes_centered_a[:, :3] = 0
        boxes_centered_a[:, 6] = 0
        boxes_centered_b[:, :3] = 0
        boxes_centered_b[:, 6] = 0
        scale_score = self.compute_iou(boxes_centered_a, boxes_centered_b)
        scale_score[scale_score < 0] = 0
        return scale_score

    def get_similarity_matrix(self, base_det_boxes, base_det_labels,
                              pert_det_boxes, pert_det_labels, pert_det_scores):
        """
        compute similarity score between the base detections in the full
        point cloud and the detections in the perturbed samples

        Parameters
        ----------
        base_det_boxes : (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : (K)
            class labels of detected objects in full pcl
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the perturbed samples of the batch
        pert_det_labels : ndarray (L)
            labels of all L detections in the perturbed samples of the batch
        pert_det_scores : ndarray (L)
            scores of all L detections in the perturbed samples of the batch
        Returns
        -------
        sim_scores : ndarray (K, L)
            similarity score between all K detections in the full pcl and
            the L detections in the perturbed samples within the batch
        """
        # ou: 以下各项属性得分都是以[base,pert]尺寸的形式给出
        # similarity score is only greater zero if boxes overlap
        s_overlap = self.compute_iou(base_det_boxes, pert_det_boxes) > 0
        s_overlap = s_overlap.astype(np.float32)

        # similarity score is only greater zero for boxes of same class
        s_class = base_det_labels[:, None] == pert_det_labels[None, :]
        s_class = s_class.astype(np.float32)

        # confidence score is directly used (see paper)
        s_conf = np.repeat(pert_det_scores[None, :], base_det_boxes.shape[0], axis=0)

        s_transl = self.compute_translation_score(base_det_boxes, pert_det_boxes)

        s_orient = self.compute_orientation_score(base_det_boxes, pert_det_boxes, ind=6)

        s_scale = self.compute_scale_score(base_det_boxes, pert_det_boxes)

        # sim_scores = s_overlap * s_conf * s_transl * s_orient * s_scale * s_class

        valid = s_overlap * s_class
        properties_score = [s_conf, s_transl, s_scale, s_orient]

        if base_det_boxes.shape[1] > 7:
            s_pitch = self.compute_orientation_score(base_det_boxes, pert_det_boxes, ind=7)
            s_roll = self.compute_orientation_score(base_det_boxes, pert_det_boxes, ind=8)
            properties_score.append(s_pitch)
            properties_score.append(s_roll)
        return valid, properties_score

    def compute_attribution_maps(self, pcl, base_det_boxes, base_det_labels,
                                 batch_size, num_workers):
        """
        attribution map computation for each base detection

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed full point cloud (x, y, z, intensity)
        base_det_boxes : ndarray (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : ndarray (K)
            class labels of detected objects in full pcl
        batch_size : int
            batch_size during AM computation
        num_workers : int
            number of dataloader workers

        Returns
        -------
        attr_maps : ndarray (K, N)
            attribution scores for all K detected base objects and all N points
        """

        # [x,y,z,dx,dy,dz,rz,ry,rx] attr_map[boxes,points,channels(conf,trans,scale,rx,ry,rz)]
        # [x,y,z,dx,dy,dz,rz], attr_map[boxes,points,channels(conf,trans,scale,heading)]
        channels = 6 if base_det_boxes.shape[1] > 7 else 4

        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0], channels))
        sampling_map = np.zeros(pcl.shape[0])

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        dataloader = DataLoader(
            occam_inference_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False,
            collate_fn=occam_inference_dataset.collate_batch, drop_last=False,
            sampler=None, timeout=0
        )

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)

        with torch.no_grad():
            for i, batch_dict in enumerate(dataloader):

                load_data_to_gpu(batch_dict)
                pert_pred_dicts, _ = self.model.forward(batch_dict)

                pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids = \
                    self.merge_detections_in_batch(pert_pred_dicts)

                valid, score_list = self.get_similarity_matrix(
                    base_det_boxes, base_det_labels,
                    pert_det_boxes, pert_det_labels, pert_det_scores)

                scores = np.stack(score_list, axis=-1) * valid[..., None]
                # scores = valid * np.stack(score_list, axis=-1).prod(axis=-1)

                cur_batch_size = len(pert_pred_dicts)
                for j in range(cur_batch_size):
                    cur_mask = batch_dict['mask'][j, :].cpu().numpy()
                    sampling_map += cur_mask

                    batch_sample_mask = batch_ids == j
                    if np.sum(batch_sample_mask) > 0:
                        max_score = np.max(scores[:, batch_sample_mask, :], axis=1)
                        # [obj,1,channels] * [pts,1]
                        attr_maps += max_score[:, None, :] * cur_mask[None, :, None]

                progress_bar.update(n=cur_batch_size)

        progress_bar.close()

        # normalize using occurrences
        attr_maps[:, sampling_map > 0, :] /= sampling_map[sampling_map > 0][:, None]
        # attr_maps = attr_maps.sum(axis=0)
        return attr_maps

    def visualize_attr_map(self, points, box, attr_map, draw_origin=True):
        turbo_cmap = plt.get_cmap('turbo')
        attr_map_scaled = attr_map - attr_map.min()
        attr_map_scaled /= attr_map_scaled.max()
        color = turbo_cmap(attr_map_scaled)[:, :3]  # rgba -> rgb

        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 4.0
        vis.get_render_option().background_color = np.ones(3) * 0.25

        if draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)

        if box.shape[0] > 7:
            rot_mat = open3d.geometry.get_rotation_matrix_from_xyz(box[6:9][::-1, None])
        else:
            rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)
        vis.add_geometry(bb)

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(color)
        vis.add_geometry(pts)

        vis.run()
        vis.destroy_window()
