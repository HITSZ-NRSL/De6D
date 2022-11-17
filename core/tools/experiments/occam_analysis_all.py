import io as sysio
import os
import sys
from pathlib import Path

path = Path("experiments/demo_data/gazebo/upslope/velodyne_points/data")
it = list(path.iterdir())
it.sort()
for p in it[:200:10]:
    cmd = f"python experiments/occam_analysis.py " \
          f"--occam_cfg_file cfgs/occam_configs/kitti.yaml " \
          f"--model_cfg_file cfgs/slopedkitti_models/det6d_pitch_car.yaml  " \
          f"--ckpt models/det6d_pitch_car_slopeaug01_80.pth " \
          f"--source_file_path {p}  " \
          f"--nr_it 10000"
    os.system(cmd)