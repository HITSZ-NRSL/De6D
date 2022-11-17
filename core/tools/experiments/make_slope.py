import argparse
import glob
import os
from pathlib import Path

import numpy

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
from pcdet.datasets.augmentor.augmentor_utils import random_global_make_slope
from pcdet.utils import box_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--smooth', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--postprocess', action='store_true', default=False)
    parser.add_argument('--dist_mean', type=float, default=None, required=False)
    parser.add_argument('--dist_var', type=float, default=None, required=False)
    parser.add_argument('--angle_mean', type=float, default=None, required=False)
    parser.add_argument('--angle_var', type=float, default=None, required=False)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


point_cloud_random_make_slope = random_global_make_slope


def main():
    args, cfg = parse_config()
    aug_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False
    )
    single_frame = args.data_path is not ''
    viz, save, smooth, postprocess, force = args.viz, args.save, args.smooth, args.postprocess, args.force
    params = [args.dist_mean, args.dist_var, args.angle_mean, args.angle_var]
    params = params if None not in params else [20, 10, 4, 24]

    print(f"====== parameters ======\n"
          f"single frame: {single_frame}\n"
          f"visualization: {viz}\n"
          f"save to file: {save}\n"
          f"force rewrite: {save}\n"
          f"smooth slope: {smooth}\n"
          f"postprocess: {postprocess}\n"
          f"parameter: \n"
          f"  dist_mean: {params[0]}\n"
          f"  dist_var: {params[1]}\n"
          f"  angle_mean: {params[2]}\n"
          f"  angle_var: {params[3]}")

    if single_frame is False:
        data_path = Path(aug_dataset.root_path).parent
        dataset_path = data_path / 'slopedkittis' / f"slopedkitti_{params[0]}_{params[1]}_{params[2]}_{params[3]}"
        assert not dataset_path.exists() or force, f'{dataset_path.name} already exists. Append arg --force to rewrite.'
        dataset_split_path = dataset_path / "training"
        print(f"====== paths ======\n"
              f"dataset root: {data_path}\n"
              f"output dataset: {dataset_path}\n"
              f"output split path: {dataset_split_path}")

        if save:
            slope_planes_dir = dataset_split_path / 'slope_planes'
            slope_points_dir = dataset_split_path / 'velodyne'
            slope_labels_dir = dataset_split_path / 'label_2'
            slope_planes_dir.mkdir(parents=True, exist_ok=True)
            slope_points_dir.mkdir(parents=True, exist_ok=True)
            slope_labels_dir.mkdir(parents=True, exist_ok=True)
            print(f"====== create directory ======\n"
                  f"planes: {slope_planes_dir}\n"
                  f"points: {slope_points_dir}\n"
                  f"labels: {slope_labels_dir}")

            calib_dir_out = (dataset_split_path / "calib").absolute()
            calib_dir_in = (data_path / "kitti/training/calib").absolute()
            image_2_dir_out = (dataset_split_path / "image_2").absolute()
            image_2_dir_in = (data_path / "kitti/training/image_2").absolute()
            image_3_dir_out = (dataset_split_path / "image_3").absolute()
            image_3_dir_in = (data_path / "kitti/training/image_3").absolute()
            planes_dir_out = (dataset_split_path / "planes").absolute()
            planes_dir_in = (data_path / "kitti/training/planes").absolute()
            try:
                os.unlink(calib_dir_out)
                os.unlink(image_2_dir_out)
                os.unlink(image_3_dir_out)
                os.unlink(planes_dir_out)
            except:
                pass

            os.symlink(calib_dir_in, calib_dir_out)
            os.symlink(image_2_dir_in, image_2_dir_out)
            os.symlink(image_3_dir_in, image_3_dir_out)
            os.symlink(planes_dir_in, planes_dir_out)
            print(f"====== craete symlink ======\n"
                  f"{calib_dir_in} -> {calib_dir_out}\n"
                  f"{image_2_dir_in} -> {image_2_dir_out}\n"
                  f"{image_3_dir_in} -> {image_3_dir_out}\n"
                  f"{planes_dir_in} -> {planes_dir_out}")

            os.system(f"cp -r {(data_path / 'kitti/ImageSets').__str__()}  {dataset_path.__str__()}")

        print(f"====== processing ======")
        params[2], params[3] = np.deg2rad(params[2:4])
        for i, data_dict in enumerate(aug_dataset):
            frame_id = data_dict['frame_id']
            print(f'{aug_dataset.root_split_path}.{frame_id}', end=' - ')
            calib = data_dict['calib']
            points = aug_dataset.get_lidar(frame_id)

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

            new_boxes, new_points, rp, ra = random_global_make_slope(gt_boxes, points, smooth=smooth, params=params)

            print(f"points: {new_points.shape} - boxes: {new_boxes.shape}")

            if save:
                slope_points_path = slope_points_dir / (frame_id + '.bin')
                slope_planes_path = slope_planes_dir / (frame_id + '.txt')
                slope_labels_path = slope_labels_dir / (frame_id + '.txt')

                with open(slope_points_path.__str__(), 'w') as f:
                    new_points.tofile(f)
                    print(f"save points: {slope_points_path}")

                with open(slope_planes_path.__str__(), 'w') as f:
                    print('%f %f %f\n'
                          '%f %f %f' % (rp[0], rp[1], rp[2], ra[0], ra[1], ra[2]),
                          file=f)
                    print(f"save params: {slope_planes_path}")

                with open(slope_labels_path.__str__(), 'w') as f:
                    # 'xyz' -> box[6:9]
                    # 'XYZ' -> box[6:9][::-1]
                    for i, label in enumerate(labels):
                        if label.cls_type != 'DontCare':
                            boxes = new_boxes[i, :]
                            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes[None, ...], calib)
                            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                                pred_boxes_camera, calib, image_shape=data_dict['image_shape']
                            )
                            label.box2d = pred_boxes_img[0, :]
                            label.l = pred_boxes_camera[0, 3:4]
                            label.h = pred_boxes_camera[0, 4:5]
                            label.w = pred_boxes_camera[0, 5:6]
                            label.loc = pred_boxes_camera[0, 0:3]
                            label.ry = pred_boxes_camera[0, 6]
                            label_str = label.to_kitti_format() + ' %f %f' % (boxes[7], boxes[8])
                        else:
                            label_str = label.to_kitti_format() + ' %f %f' % (-10, -10)
                        print(label_str, file=f)
                    print(f"save labels: {slope_labels_path}")

            if viz:
                print("visualization ...")
                draw_points = new_points
                draw_boxes = new_boxes
                ## viz
                vis = open3d.visualization.Visualizer()
                vis.create_window()
                vis.get_render_option().point_size = 2.0
                vis.get_render_option().background_color = np.ones(3) * 0.3
                axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                vis.add_geometry(axis_pcd)

                pts = open3d.geometry.PointCloud()
                pts.points = open3d.utility.Vector3dVector(draw_points[:, :3])
                pts.colors = open3d.utility.Vector3dVector(np.zeros((draw_points.shape[0], 3)))
                vis.add_geometry(pts)

                mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                mesh_sphere.translate(rp)
                mesh_sphere.compute_vertex_normals()
                vis.add_geometry(mesh_sphere)
                vis = V.draw_box(vis, draw_boxes, (0, 0, 1))

                vis.run()
                vis.destroy_window()

        # postprocess
        if save and postprocess:
            print(f"====== postprocess ======\n")
            try:
                current_slopedkitti = (data_path / "slopedkitti").absolute()
                os.unlink(current_slopedkitti)
                print(f"unlink : {current_slopedkitti}")
            except:
                pass
            os.symlink(dataset_path.absolute(), current_slopedkitti)
            print(f"symlink : {dataset_path.absolute()} -> {current_slopedkitti}")
            print(f"database generation ...")
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            os.system("cd .. && python -m "
                      "pcdet.datasets.slopedkitti.kitti_dataset "
                      "create_kitti_infos "
                      "tools/cfgs/dataset_configs/slopedkitti_dataset.yaml")
    else:
        file_name = Path(args.data_path)
        assert file_name.is_file()

        print(f"====== processing ======")
        print(file_name, end=' - ')
        points = np.fromfile(str(file_name), dtype=np.float32).reshape(-1, 4)
        params[2], params[3] = np.deg2rad(params[2:4])

        new_boxes, new_points, rp, ra = random_global_make_slope(np.zeros([0, 9]), points, smooth=smooth, params=params)
        print(f"points: {new_points.shape}")
        if save:
            save_file = Path('experiments/demo_data') / file_name.stem
            save_file = save_file.with_suffix('.bin')
            with open(save_file, 'w') as f:
                new_points.tofile(f)
                print(f"save points: {save_file}")
        if viz:
            V.draw_scenes(points=new_points)
            if not OPEN3D_FLAG:
                mlab.show(stop=True)


if __name__ == '__main__':
    main()
