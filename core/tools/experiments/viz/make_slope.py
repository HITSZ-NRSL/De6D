import argparse
import glob
import os
from pathlib import Path

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
from scipy.spatial.transform import Rotation
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import box_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--dist_mean', type=float, default=None, required=False)
    parser.add_argument('--dist_var', type=float, default=None, required=False)
    parser.add_argument('--angle_mean', type=float, default=None, required=False)
    parser.add_argument('--angle_var', type=float, default=None, required=False)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def point_cloud_random_make_slope(gt_boxes, gt_points, params=None, rotate_point=None, rotate_angle=None, smooth=False):
    from pcdet.utils import common_utils
    from scipy.spatial.transform import Rotation
    def random(n=1):
        return (np.random.random(n) - 0.5) * 2

    ##########################################
    # get rotate point and rotate angle
    ##########################################
    assert params is not None

    dist_mean, dist_var, angle_mean, angle_var = params
    points = gt_points.copy()

    if rotate_point is None:
        mean, var = np.array([dist_mean, 0]), np.array([dist_var, 0])
        polar_pos = mean + random(2) * var
        rotate_point = np.array([polar_pos[0] * np.cos(polar_pos[1]), polar_pos[0] * np.sin(polar_pos[1]), 0])

    x0, y0 = rotate_point[0], rotate_point[1]
    if rotate_angle is None:
        if x0 > dist_mean:
            angle_mean = angle_mean + angle_var / 2
            angle_var = angle_var / 2
        if x0 < dist_mean:
            angle_mean = angle_mean - angle_var / 2
            angle_var = angle_var / 2

        mean, var = angle_mean, angle_var
        k0 = y0 / x0
        k1 = -1 / (k0 + 1e-6)
        v = np.array([x0 - 0, y0 - (-x0 * k1 + y0), 0])
        v /= np.linalg.norm(v)
        angle = mean + random() * var
        v *= angle
        direction = np.sign(np.cross(rotate_point, v)[2])
        # v *= -1 if direction > 0 else 1
        rotate_angle = v

    ##########################################
    # apply sloped-aug in smooth condition
    ##########################################
    if smooth:
        radius, bins = rotate_point[0] / np.abs(rotate_angle[1]), 8
        alpha = rotate_angle[1]
        dist = rotate_point[0]
        for theta in np.linspace(0, alpha, bins):
            delta = alpha / bins
            center = np.array([dist, 0, radius])
            rotate_point = center + np.array([-radius * np.sin(theta), 0, -radius * np.cos(theta)])
            rotate_angle = np.array([0, delta, 0])
            gt_boxes, points, rotate_point, rotate_angle = point_cloud_random_make_slope(
                gt_boxes, points,
                params=(20, 12, *np.deg2rad([21, 7])),
                rotate_angle=rotate_angle,
                rotate_point=rotate_point)
    else:
        k = rotate_angle[1] / (rotate_angle[0] + 1e-6)
        sign = np.sign(k * (0 - x0) + y0 - 0)
        in_plane_mask = np.sign(k * (points[:, 0] - x0) + y0 - points[:, 1]) != sign
        slope_points = points[in_plane_mask]
        slope_points[:, 0:3] -= rotate_point
        rot = Rotation.from_rotvec(rotate_angle).as_matrix()
        slope_points[:, 0:3] = (slope_points[:, 0:3].dot(rot.T))
        slope_points[:, 0:3] += rotate_point
        points[in_plane_mask] = slope_points
        mask = in_plane_mask.copy()
        # gt_box
        if gt_boxes.shape[1] < 9:
            gt_boxes = np.concatenate((gt_boxes, np.zeros([gt_boxes.shape[0], 2])), axis=1)
        in_plane_mask = np.sign(k * (gt_boxes[:, 0] - x0) + y0 - gt_boxes[:, 1]) != sign  # box position mask
        slope_box = gt_boxes[in_plane_mask]
        slope_box[:, :3] -= rotate_point
        slope_box[:, :3] = (slope_box[:, :3].dot(rot.T))
        slope_box[:, :3] += rotate_point
        gt_boxes[in_plane_mask] = slope_box
        euler = Rotation.from_rotvec(rotate_angle).as_euler('XYZ')
        # pts × ((RxRy)Rz).T
        # boxes(9)[x, y, z, dx, dy, dz, rz(, ry, rx)[Rot=RxRyRz]]
        gt_boxes[in_plane_mask, 7] += euler[1]  # y
        gt_boxes[in_plane_mask, 8] += euler[0]  # y
        gt_boxes[:, 6:9] = common_utils.limit_period(
            gt_boxes[:, 6:9], offset=0.5, period=2 * np.pi
        )
    return gt_boxes, points, rotate_point, rotate_angle, mask


def main():
    args, cfg = parse_config()
    aug_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False
    )

    smooth_between = False
    params = [[25, 0],
              [25, 0]]

    data_path = Path(aug_dataset.root_path).parent
    dataset_path = data_path / f"slopekitti_{params[0][0]}_{params[0][1]}_{params[1][0]}_{params[1][1]}"
    dataset_split_path = dataset_path / "training"
    print(f"data path: {data_path}")
    print(f"dataset path: {dataset_path}")
    print(f"dataset split path: {dataset_split_path}")
    data_list = [222]
    for i, sample_idx in enumerate(data_list):
        data_dict = aug_dataset[sample_idx]
        frame_id = data_dict['frame_id']
        print('process: {}.{}'.format(aug_dataset.root_split_path, frame_id))
        calib = data_dict['calib']
        points = aug_dataset.get_lidar(frame_id)

        img_shape = aug_dataset.kitti_infos[i]['image']['image_shape']
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = aug_dataset.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        def get_boxes(scene):
            obj_list = aug_dataset.get_label(scene)
            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([obj.score for obj in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            loc_lidar[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar
            return obj_list, annotations['gt_boxes_lidar']

        labels, gt_boxes = get_boxes(frame_id)
        print(f"boxes: {gt_boxes.shape}")

        new_boxes, new_points, rp, ra, mask = point_cloud_random_make_slope(
            gt_boxes, points, smooth=smooth_between,
            params=(params[0][0], params[0][1], *np.deg2rad([params[1][0], params[1][1]])))

        print(f"points: {new_points.shape}, boxes: {new_boxes.shape}")

        def wait():
            nonlocal stop
            stop = True
            while stop:
                vis.poll_events()
                vis.update_renderer()

        def key_action_callback(vis, action, mods):
            if action == 0:
                nonlocal stop
                stop = False
                print(f'key_action_callback: {stop}.')
        vp = "experiments/viz/viewpoints/pv1.json"
        stop = True
        draw_points = new_points
        draw_boxes = new_boxes
        ## viz
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.get_render_option().point_size = 3.0
        vis.register_key_action_callback(32, key_action_callback)
        vis.get_render_option().background_color = np.ones(3) * 1
        print("====== press 'Space' to continue ======")
        # axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # vis.add_geometry(axis_pcd)

        ################################################################################################################
        # raw point cloud
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * 0.3)
        vis.add_geometry(pts)

        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=1)
        mesh_sphere.translate(rp)
        mesh_sphere.compute_vertex_normals()
        vis.add_geometry(mesh_sphere)

        V.draw_box(vis, gt_boxes, color=[1, 0, 0], width=5)

        mesh_sphere = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.5, cone_radius=0.75,
            cylinder_height=36.0, cone_height=4.0,
            resolution=100, cylinder_split=4, cone_split=1
        )
        mesh_sphere.paint_uniform_color([0, 0, 1])
        mesh_sphere.translate(rp)
        mesh_sphere.translate([0, 0, -21])
        from scipy.spatial.transform.rotation import Rotation as Rot
        mesh_sphere.rotate(Rot.from_euler('zyx', [0, 0, - np.pi / 2]).as_matrix())
        mesh_sphere.compute_vertex_normals()
        vis.add_geometry(mesh_sphere)
        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # raw point cloud and box
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * 0.3)

        colors = np.asarray(pts.colors)
        colors[mask] = np.array([[0, 0.6, 0]])
        vis.add_geometry(pts)

        V.draw_box(vis, gt_boxes, color=[1, 0, 0], width=5)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # separate 0
        p1 = points[np.logical_not(mask)]
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(p1[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((p1.shape[0], 3)) * 0.3)
        vis.add_geometry(pts)

        wait()
        vis.clear_geometries()

        ################################################################################################################
        # separate 1
        p1 = points[mask]
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(p1[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((p1.shape[0], 3)) * 0.3)
        colors = np.asarray(pts.colors)
        colors[...] = np.array([[0, 0.6, 0]])
        vis.add_geometry(pts)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # raw box
        V.draw_box(vis, gt_boxes, color=[1, 0, 0], width=5)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # new separate 0
        p1 = new_points[np.logical_not(mask)]
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(p1[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((p1.shape[0], 3)) * 0.3)
        vis.add_geometry(pts)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # new separate 1
        p1 = new_points[mask]
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(p1[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((p1.shape[0], 3)) * 0.3)

        colors = np.asarray(pts.colors)
        colors[...] = np.array([[0, 0.6, 0]])
        vis.add_geometry(pts)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        # new box
        V.draw_box(vis, new_boxes, color=[1, 0, 0], width=5)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(draw_points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((draw_points.shape[0], 3)) * 0.3)

        colors = np.asarray(pts.colors)
        colors[mask] = np.array([[0, 0.6, 0]])
        vis.add_geometry(pts)

        # mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        # mesh_sphere.translate(rp)
        # mesh_sphere.compute_vertex_normals()
        # vis.add_geometry(mesh_sphere)

        vis = V.draw_box(vis, draw_boxes, (1, 0, 0), width=5)

        params = open3d.io.read_pinhole_camera_parameters(vp)
        vc = vis.get_view_control()
        vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        wait()
        vis.clear_geometries()

        ################################################################################################################

        ################################################################################################################
        vis.destroy_window()


if __name__ == '__main__':
    main()
