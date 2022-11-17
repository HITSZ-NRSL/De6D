import pickle
import numpy as np
import copy
import argparse
import warnings

warnings.filterwarnings("ignore")
from pcdet.datasets.slopedkitti.kitti_object_eval_python import eval as slopedkitti_eval

dt_result_path = "../output/slopedkitti_models/det6d_car/default/eval/epoch_80/val/default/result.pkl"
gt_result_path = "../data/slopedkitti/kitti_infos_val.pkl"

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--file', type=str, default=dt_result_path, help='specify the result.pkl')
args = parser.parse_args()

with open(args.file, 'rb') as f:
    dt_annos = copy.deepcopy(pickle.load(f))

with open(gt_result_path, 'rb') as f:
    infos = pickle.load(f)
    gt_annos = [copy.deepcopy(info['annos']) for info in infos]

print(f"num_scenes: {len(dt_annos)}")
print(f"num_dt_obj: {np.array([cnt['name'].shape[0] for cnt in dt_annos]).sum()}")
print(f"num_gt_obj: {np.array([cnt['name'].shape[0] for cnt in gt_annos]).sum()}")
print(f"num_dt_Car: {np.array([(np.array(np.array(cnt['name']) == 'Car')).sum() for cnt in dt_annos]).sum()}")
print(f"num_gt_Car: {np.array([(np.array(cnt['name']) == 'Car').sum() for cnt in gt_annos]).sum()}")
result, _ = slopedkitti_eval.get_slopedkitti_eval_result(gt_annos, dt_annos, current_classes=['Car'])
print(result)
