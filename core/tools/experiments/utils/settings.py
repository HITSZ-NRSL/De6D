# models_active = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
models_active = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]  # for eval on various slopedkitti
datasets = [
    "../data/slopedkittis/slopedkitti",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_-10.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_-5.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_0.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_5.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_10.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_15.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_20.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_25.0_0.0",
    # "../data/slopedkittis/slopedkitti_22.0_8.0_30.0_0.0",
]
kitti_serial_datasets = [
    "experiments/demo_data/kitti_serial/2011_09_26_drive_0001_sync/velodyne_points/data"
    # "experiments/demo_data/kitti_serial/2011_09_26_drive_0059_sync/velodyne_points/data"
    # "experiments/demo_data/kitti_serial/2011_09_26_drive_0093_sync/velodyne_points/data"
    # "experiments/demo_data/kitti_serial/2011_09_26_drive_0095_sync/velodyne_points/data"
    # "experiments/demo_data/kitti_serial/2011_09_26_drive_0104_sync/velodyne_points/data"
    # "experiments/demo_data/kitti_serial/2011_09_26_drive_0106_sync/velodyne_points/data"
]
gazebo_serial_datasets = [
    "experiments/demo_data/gazebo/upslope/velodyne_points/data",
    "experiments/demo_data/gazebo/downslope/velodyne_points/data",
]
cfgs = [
    "cfgs/kitti_models/det6d_car.yaml",
    "cfgs/kitti_models/det6d_pitch_car.yaml",
    "cfgs/kitti_models/3dssd_car.yaml",
    "cfgs/kitti_models/3dssd_sasa_car.yaml",
    "cfgs/kitti_models/centerpoint_nms.yaml",
    "cfgs/kitti_models/IA-SSD.yaml",
    "cfgs/kitti_models/PartA2_free.yaml",
    "cfgs/kitti_models/pointpillar.yaml",
    "cfgs/kitti_models/pointrcnn.yaml",
    "cfgs/kitti_models/pointrcnn_slopeaug.yaml",
    "cfgs/kitti_models/pv_rcnn.yaml",
    "cfgs/kitti_models/second.yaml",
    "cfgs/kitti_models/voxel_rcnn_car.yaml",
]
slopedkitti_cfgs = [
    "cfgs/slopedkitti_models/det6d_car.yaml",
    "cfgs/slopedkitti_models/det6d_pitch_car.yaml",
    "cfgs/slopedkitti_models/3dssd_car.yaml",
    "cfgs/slopedkitti_models/3dssd_sasa_car.yaml",
    "cfgs/slopedkitti_models/centerpoint_nms.yaml",
    "cfgs/slopedkitti_models/IA-SSD.yaml",
    "cfgs/slopedkitti_models/PartA2_free.yaml",
    "cfgs/slopedkitti_models/pointpillar.yaml",
    "cfgs/slopedkitti_models/pointrcnn.yaml",
    "cfgs/slopedkitti_models/pointrcnn_slopeaug.yaml",
    "cfgs/slopedkitti_models/pv_rcnn.yaml",
    "cfgs/slopedkitti_models/second.yaml",
    "cfgs/slopedkitti_models/voxel_rcnn_car.yaml",
]
ckpts = [
    "models/det6d_car_slopeaug01_80.pth",
    "models/det6d_pitch_car_slopeaug01_80.pth",
    "models/3dssd_bs8x2-checkpoint_epoch_80.pth",
    "models/3dssd_sasa_car_79.pth",
    "models/centerpoint_kitti_80.pth",
    "models/IA-SSD.pth",
    "models/PartA2_free_7872.pth",
    "models/pointpillar_7728.pth",
    "models/pointrcnn_7870.pth",
    "models/pointrcnn_slope_80.pth",
    "models/pv_rcnn_8369.pth",
    "models/second_7862.pth",
    "models/voxel_rcnn_car_84.54.pth",
]


def filtered_by_mask(x, mask):
    return [i for i, valid in zip(x, mask) if valid != 0]


for (i, cfg), valid, scfg, ckpt in zip(enumerate(cfgs), models_active, slopedkitti_cfgs, ckpts):
    print(f"[{'v' if valid else 'x'}] {i} {cfg} {scfg} {ckpt}")

cfgs = filtered_by_mask(cfgs, models_active)
slopedkitti_cfgs = filtered_by_mask(slopedkitti_cfgs, models_active)
ckpts = filtered_by_mask(ckpts, models_active)
