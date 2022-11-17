import copy
import time

import numpy
import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ..model_utils import model_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.box_utils import points_in_boxes3d
from scipy.spatial.transform import Rotation
from ...utils import loss_utils
from ...utils.common_utils import TimeMeasurement


def point_r_cnn_hook(input_dict=None):
    global hook_dict
    if input_dict is not None:
        hook_dict = {'point_cls_scores': copy.deepcopy(input_dict['point_cls_scores']),
                     'cls_preds': copy.deepcopy(input_dict['batch_cls_preds']),
                     'point_cls_preds': copy.deepcopy(input_dict['point_cls_preds']),
                     'points': copy.deepcopy(input_dict['points']),
                     'point_box_preds': copy.deepcopy(input_dict['batch_box_preds']),
                     'point_box_preds_raw': copy.deepcopy(input_dict['point_box_preds_raw']),
                     'point_rot_cls_preds': copy.deepcopy(input_dict['point_rot_cls_preds'])}
    else:
        return hook_dict


class PointHeadBox6D(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        self.rot_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.ROT_CLS_FC,
            input_channels=input_channels,
            output_channels=2  # 有否pitch
        )
        self.rot_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.ROT_REG_FC,
            input_channels=input_channels,
            output_channels=3
        )
        self.rot_reg_loss_func = F.mse_loss
        self.add_module(
            'rot_cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=1)
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """

        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        with TimeMeasurement("PointHeadBox6D.AssignTarget.Enlarge") as tx:
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
            )
        extend_gt_boxes = extend_gt_boxes.view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets_(points=point_coords,
                                                  gt_boxes=gt_boxes,
                                                  extend_gt_boxes=extend_gt_boxes)

        return targets_dict

    def assign_stack_targets_(self, points, gt_boxes, extend_gt_boxes=None, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        # 输入检查
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] in [8, 10], 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] in [8, 10], \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        t0 = TimeMeasurement("PointHeadBox6D.AssignTarget.StackTargets")
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()  # box分类标签
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8))  # box回归标签
        point_rot_cls_labels = gt_boxes.new_zeros((points.shape[0])).long()  # 角度分类标签
        point_rot_reg_labels = gt_boxes.new_zeros((points.shape[0], 3))  # 角度回归标签

        for k in range(batch_size):
            bs_mask = (bs_idx == k)  # maks:点云中属于此batch

            # ** box分类标签 **
            points_single = points[bs_mask][:, 1:4]  # 本batch中的点云
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())  # 本batch点云的分类标签

            # 点是否在 box
            box_idxs_of_pts = points_in_boxes3d(points_single, gt_boxes[k:k + 1, ...].squeeze(dim=0)).long()
            # 点是否在 enlarge box 中
            extend_box_idxs_of_pts = points_in_boxes3d(points_single,
                                                       extend_gt_boxes[k:k + 1, ...].squeeze(dim=0)).long()

            # 前景点 >0：如果点在任何一个box中
            box_fg_flag = (box_idxs_of_pts >= 0)  # 点云是否在box内
            fg_flag = box_fg_flag  # 在box内的点云为正样本
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]  # 正点云对应的gtbox
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()

            # 忽略点 -1：如果点在enlarge box 而不在 box 中
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            point_cls_labels_single[ignore_flag] = -1
            # 设置本batch的点云分类标签
            point_cls_labels[bs_mask] = point_cls_labels_single

            # 至少有一个box中有点云
            if gt_box_of_fg_points.shape[0] > 0:
                # ** box回归标签 **
                # fg点由box确定回归目标，其余点回归目标为0
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :7], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

                # ** 角度回归标签 **
                # fg点由box确定回归目标，其余点回归目标为0
                point_rot_reg_labels_single = point_rot_reg_labels.new_zeros((bs_mask.sum(), 3))
                # R=Rx(8)Ry(7)Rz(6)
                axis_angle = Rotation.from_euler('zyx', gt_box_of_fg_points[:, 6:9].cpu().numpy()).as_rotvec()
                point_rot_reg_labels_single[fg_flag] = torch.tensor(axis_angle).float().to(point_rot_reg_labels.device)
                point_rot_reg_labels[bs_mask] = point_rot_reg_labels_single

                # ** 角度分类标签 **
                # 非fg点为-1，非斜坡fg点为0，斜坡fg点为1
                point_rot_cls_labels_single = point_rot_cls_labels.new_zeros(bs_mask.sum()).fill_(-1)
                rot_fg_flag = gt_box_of_fg_points[:, 7] < -0.1
                point_rot_cls_labels_single[fg_flag] = rot_fg_flag.long()
                point_rot_cls_labels[bs_mask] = point_rot_cls_labels_single

            # points_color = torch.zeros_like(points_single)[:, 0:3] + 0.5
            # points_color[ignore_flag] = 1
            # points_color[fg_flag, 0] = 1
            # V.draw_scenes(points=points_single,
            #               gt_boxes=gt_boxes[k:k + 1, ...].squeeze(dim=0),
            #               point_colors=points_color)
            #
            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_rot_cls_labels': point_rot_cls_labels,
            'point_rot_reg_labels': point_rot_reg_labels
        }
        return targets_dict

    def get_rot_reg_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0  # (num_pts,1[True or False)
        point_rot_labels = self.forward_ret_dict['point_rot_reg_labels']  # (num_pts,3[axis-angle])
        point_rot_preds = self.forward_ret_dict['point_rot_reg_preds']  # (num_pts,3[axis-angle])

        point_loss_rot = self.rot_reg_loss_func(point_rot_preds[pos_mask, :], point_rot_labels[pos_mask, :])

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_rot = point_loss_rot * loss_weights_dict['point_rot_reg_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_rot_reg': point_loss_rot.item()})
        return point_loss_rot, tb_dict

    def get_rot_cls_layer_loss(self, tb_dict=None):
        point_rot_cls_labels = self.forward_ret_dict['point_rot_cls_labels']  # (num_pts,3[axis-angle])
        point_rot_cls_preds = self.forward_ret_dict['point_rot_cls_preds']  # (num_pts,3[axis-angle])

        positives = (point_rot_cls_labels > 0)
        negative_cls_weights = (point_rot_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = cls_weights.sum(dim=0).float()
        # 采用所有前景点数来归一化
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_rot_cls_preds.new_zeros(*list(point_rot_cls_labels.shape), 2)
        one_hot_targets.scatter_(-1,
                                 (point_rot_cls_labels * (point_rot_cls_labels >= 0).long()).unsqueeze(dim=-1).long(),
                                 1.0)

        cls_loss_src = self.rot_cls_loss_func(point_rot_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()
        loss_show = point_loss_cls.item()
        # print('\n{}, {}, {}\n'.format(negative_cls_weights.sum(), positives.sum(), loss_show))
        # if loss_show > 1 or loss_show < 1e-3:

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_rot_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_rot_cls': point_loss_cls.item(),
            'point_rot_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        point_loss_rot_reg, tb_dict_3 = self.get_rot_reg_layer_loss()
        point_loss_rot_cls, tb_dict_4 = self.get_rot_cls_layer_loss()

        point_loss = point_loss_cls + point_loss_box + point_loss_rot_reg + point_loss_rot_cls
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_3)
        tb_dict.update(tb_dict_4)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        time0 = TimeMeasurement("PointHeadBox6D")
        # 是否使用多层VSA特征concatenate但送进mlp前的特征
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']

        # 进行预测
        with TimeMeasurement("PointHeadBox6D::Pred") as timex:
            point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
            point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
            point_rot_cls_preds = self.rot_cls_layers(point_features)  # (total_points, 2)
            point_rot_reg_preds = self.rot_reg_layers(point_features)  # (total_points, 3)

        # 点云分割置信度
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        # ret_dict用于损失计算
        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds,
                    'point_rot_cls_preds': point_rot_cls_preds,
                    'point_rot_reg_preds': point_rot_reg_preds}

        # 训练期：获取预测真值(pts的分类标签(ignore enlarge gt box)、box残差真值、axis-angle真值)
        if self.training:
            with TimeMeasurement("PointHeadBox6D.AssignTarget") as t0:
                targets_dict = self.assign_targets(batch_dict)
                ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
                ret_dict['point_box_labels'] = targets_dict['point_box_labels']
                ret_dict['point_rot_cls_labels'] = targets_dict['point_rot_cls_labels']
                ret_dict['point_rot_reg_labels'] = targets_dict['point_rot_reg_labels']

        # 测试或有第二阶段：用预测值生成rois
        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            # 斜坡物体使用预测的pitch，非斜坡物体pitch设置为0
            rot_cls, rot_score = torch.max(point_rot_cls_preds, dim=1)
            rot_ng_flag = rot_cls == 0
            rot_fg_flag = rot_cls > 0
            point_rot_preds_temp = point_rot_reg_preds.clone().cpu().numpy()
            point_rot_preds_temp = Rotation.from_rotvec(point_rot_preds_temp).as_euler('zyx', degrees=False)
            point_rot_preds_temp = torch.from_numpy(point_rot_preds_temp).float().to(point_box_preds.device)

            point_rot_preds_temp[rot_ng_flag, 1] = 0  # 非斜坡物体pitch设置为0.
            point_box_preds = torch.cat((point_box_preds, point_rot_preds_temp[:, 1:]), dim=-1)  # 只用pitch和raw

            if self.predict_boxes_when_training:
                batch_dict['batch_cls_preds'] = point_cls_preds
                batch_dict['batch_box_preds'] = point_box_preds
                batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
                batch_dict['cls_preds_normalized'] = False  # cls预测结果没有sigmoid过
            else:
                # box_preds, cls_preds = self.post_processing(batch_size=batch_dict['batch_size'],
                #                                             batch_index=batch_dict['point_coords'][:, 0],
                #                                             batch_pt_box_preds=point_box_preds,
                #                                             batch_pt_cls_preds=point_cls_preds)
                batch_dict['batch_box_preds'] = point_box_preds
                batch_dict['batch_cls_preds'] = point_cls_preds
                batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
                batch_dict['cls_preds_normalized'] = False  # cls预测结果没有sigmoid过
        self.forward_ret_dict = ret_dict

        # for debug and viz
        if not self.training:
            pass
            # batch_dict['point_cls_preds'] = point_cls_preds
            # batch_dict['point_box_preds_raw'] = ret_dict['point_box_preds']
            # batch_dict['point_rot_cls_preds'] = ret_dict['point_rot_cls_preds']
            # point_r_cnn_hook(batch_dict)
        return batch_dict

    def post_processing(self, batch_size, batch_index, batch_pt_cls_preds, batch_pt_box_preds):
        nms_config = self.model_cfg.POST_PROCESSING.NMS_CONFIG
        rois = [batch_pt_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_pt_box_preds.shape[-1]))]
        roi_cls = batch_pt_box_preds.new_zeros(
            (batch_size, nms_config.NMS_POST_MAXSIZE, batch_pt_cls_preds.shape[-1]))

        for index in range(batch_size):
            batch_mask = (batch_index == index)

            box_preds = batch_pt_box_preds[batch_mask]
            cls_preds = batch_pt_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_cls[index, :len(selected)] = cls_preds[selected]
        return rois, roi_cls
