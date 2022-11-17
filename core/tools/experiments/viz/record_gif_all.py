import os
import sys
from tools.experiments.utils.settings import kitti_serial_datasets, slopedkitti_cfgs, ckpts

cfgs = slopedkitti_cfgs
datasets = kitti_serial_datasets

for dataset in datasets:
    for cfg, ckpt in zip(cfgs, ckpts):
        cmd = f"python experiments/viz/record_gif.py --cfg_file {cfg} --ckpt {ckpt} --data_path {dataset}"
        print(cmd)
        os.system(cmd)
