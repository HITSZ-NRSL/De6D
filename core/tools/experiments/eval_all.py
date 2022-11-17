import io as sysio
import os
import sys
from pathlib import Path
from utils.settings import datasets, slopedkitti_cfgs, ckpts

cfgs = slopedkitti_cfgs
for dataset in datasets:
    if 'eval' in sys.argv:
        if 'slopedkitti' in dataset:
            try:
                dataset_path = Path('../data/slopedkitti')
                os.unlink(dataset_path)
                print(f"unlink : {dataset_path}")
            except:
                pass
            os.symlink(dataset, dataset_path)
            print(f"symlink : {dataset} -> {dataset_path}")
        for cfg, ckpt in zip(cfgs, ckpts):
            cmd = f"python test.py --cfg_file {cfg} --ckpt {ckpt} --eval_tag eval_{Path(dataset).stem.__str__()} #> /dev/null 2>&1"
            print(cmd)
            os.system(cmd)

    def get_str_from_list(x):
        sstream = sysio.StringIO()
        sstream.truncate(0)
        sstream.seek(0)
        print(*x, sep=' ',file=sstream,end=' ')
        return sstream.getvalue()

    cmd = f"python experiments/utils/collect_eval_result_from_logs.py" \
          f" --cfgs {get_str_from_list(cfgs)}" \
          f" --ckpts {get_str_from_list(ckpts)}" \
          f" --datasets {get_str_from_list(datasets)}" \
          f" --save"
    print(cmd)
    os.system(cmd)
