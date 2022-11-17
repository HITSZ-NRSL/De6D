#!/home/ou/software/anaconda3/envs/openpcdet/bin/python

import os
import yaml
import glob
import time
import numpy as np
import torch
import argparse

from pathlib import Path
from easydict import EasyDict
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from pcdet.utils.box_utils import boxes3d_to_corners_3d, boxes_to_corners_3d
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

import rospy
import ros_numpy as rnp
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray


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

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--topic', type=str, default='/points', help='topic of point cloud')
    parser.add_argument('--ws', type=str, default='../../core/tools',
                        help='python current work space')

    args = parser.parse_args()

    os.chdir(args.ws)
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    return args, cfg


class NetWorks:
    def __init__(self, args, cfg):
        input_dict = EasyDict()
        input_dict.cfg_file = args.cfg_file
        input_dict.ckpt_file = args.ckpt
        input_dict.root_path = Path("")
        input_dict.score_threashold = 0.2

        # logger
        self.logger = common_utils.create_logger(rank=cfg.LOCAL_RANK)

        # build dataset and network
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=input_dict.root_path, logger=self.logger, ext='.bin')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=input_dict.ckpt_file, to_cpu=self.device == "cpu", logger=self.logger)
        self.model.to(self.device)
        self.model.eval()

    def inference(self, points):
        with torch.no_grad():
            # prepare data and load data to gpu
            input_dict = {
                'points': points,
                'frame_id': 0,
            }
            data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # inference once in batch size : 1
            pred_dicts = self.model.forward(data_dict)[0][0]

            # analysis the result
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            return pred_labels, pred_scores, pred_boxes

    def get_class_names(self):
        return cfg.CLASS_NAMES


def get_marker(bbox3d, header):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header = header
    marker.type = marker.LINE_LIST
    marker.action = marker.ADD
    marker.header.stamp = rospy.Time.now()

    # marker scale (scale y and z not used due to being linelist)
    marker.scale.x = 0.08
    # marker color
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.points = []
    corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6, 1, 4, 0, 5]
    if bbox3d.shape[-1] > 7:
        corners3d = boxes3d_to_corners_3d(bbox3d)  # (N,8,3)
    else:
        corners3d = boxes_to_corners_3d(bbox3d)
    for box_nr in range(corners3d.shape[0]):
        box3d_pts_3d_velo = corners3d[box_nr]  # (8,3)
        for corner in corner_for_box_list:
            transformed_p = np.array(box3d_pts_3d_velo[corner, 0:3])
            # transformed_p = transform_point(p, np.linalg.inv(self.Tr_velo_kitti_cam))
            p = Point()
            p.x = transformed_p[0]
            p.y = transformed_p[1]
            p.z = transformed_p[2]
            marker.points.append(p)
    marker_array.markers.append(marker)

    id = 0
    for m in marker_array.markers:
        m.id = id
        id += 1
    return marker_array


def callback(pc2, pub):
    header = pc2.header
    points = rnp.point_cloud2.pointcloud2_to_xyz_array(pc2)
    points = np.hstack((points, np.zeros([len(points), 1])))
    t1 = time.time()
    labels, scores, boxes = network.inference(points)
    t2 = time.time()
    print("detection: {} objects in {}ms".format(scores.shape[0], 1e3 * (t2 - t1)))
    # publisher
    header.stamp = rospy.Time.now()
    pc2.header = header
    pub['point'].publish(pc2)

    marker = get_marker(boxes, header)
    pub['marker'].publish(marker)


args, cfg = parse_config()
network = NetWorks(args, cfg)
rospy.init_node('openpcdet')
pub = {'point': rospy.Publisher('points', PointCloud2, queue_size=1),
       'marker': rospy.Publisher('marker', MarkerArray, queue_size=1)}
rospy.Subscriber(args.topic, PointCloud2, callback, pub)
rospy.spin()
