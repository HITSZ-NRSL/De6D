import os
import sys
from tools.experiments.utils.settings import slopedkitti_cfgs,ckpts
cfgs=slopedkitti_cfgs
for cfg, ckpt in zip(cfgs, ckpts):
    cmd = f"python experiments/resultss.py --cfg_file {cfg} --ckpt {ckpt}"
    print(cmd)
    os.system(cmd)
