import os
import sys
from utils.settings import cfgs,ckpts

for cfg, ckpt in zip(cfgs, ckpts):
    os.system(f"python demo.py --cfg {cfg} --ckpt {ckpt} --data_path {sys.argv[1]}")
