import pickle
import sys
import re
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfgs', nargs="*", type=str, default=None)
parser.add_argument('--ckpts', nargs="*", type=str, default=None)
parser.add_argument('--datasets', nargs="*", type=str, default=None)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--file', type=str, default=None)


def prefix(level):
    return ' ' * int(level) * 2


def performance_handle(data_list, eval_dict):
    def metric_handle():
        cls = metric_block[0][:-2]
        print(f"{prefix(2)}decode metric: {cls}")

        bbox = [float(m) for m in metric_block[2][8:-1].split(sep=',')]
        bev = [float(m) for m in metric_block[3][8:-1].split(sep=',')]
        d3d = [float(m) for m in metric_block[4][8:-1].split(sep=',')]
        aos = [float(m) for m in metric_block[5][8:-1].split(sep=',')]
        oo3d = [float(m[:-1].split(sep=' ')[-1]) for m in metric_block[6:metric_block_size - 1]]
        eval_dict[cls] = {'bbox': bbox, 'bev': bev, '3d': d3d, 'aos': aos, 'oo3d': oo3d}

    result_offset = 10
    metric_block_size = 12
    result_strs = data_list[result_offset:]
    for metric_i in range(len(result_strs[::metric_block_size])):
        metric_block = result_strs[metric_i * metric_block_size:(metric_i + 1) * metric_block_size]
        if metric_block.__len__() == metric_block_size:
            metric_handle()


def analysis(log_root):
    data_dict = {}
    for cfg, ckpt in zip(cfgs, ckpts):
        cfg_name, ckpt_name = Path(cfg).stem, Path(ckpt).stem
        epoch = re.findall(r'\d+', ckpt_name)
        epoch = epoch[-1] if epoch.__len__() > 0 else 'no_number'  # no_number

        log_dir = log_root / cfg_name / 'default/eval' / f'epoch_{epoch}' / 'val'
        print(f"--------------------------------\n"
              f"[{'v' if log_dir.exists() else 'x'}] log directory: {log_dir}")

        data_dict[f'{cfg_name}'] = {}
        for eval_tag in eval_tags:
            print(f"{prefix(0)}handle {eval_tag} ...", end='\n')
            log_dir_eval_tag = log_dir / eval_tag
            log_files = []
            for log_file in log_dir_eval_tag.iterdir():
                if log_file.suffix == '.txt':
                    log_files.append(log_file.stem.__str__() + '.txt')
            log_file = log_dir_eval_tag / max(log_files)
            print(f"{prefix(1)}read {log_file}")
            with open(log_file.__str__(), 'r') as f:
                lines = f.readlines()
                from_which = None
                for i, line in enumerate(lines):
                    if f"*************** Performance of EPOCH {epoch} *****************" in line:
                        print(f"{prefix(1)}find eval result")
                        from_which = i
                        break
                if from_which is None:
                    break
            data_dict[f'{cfg_name}'][eval_tag] = {}
            performance_handle(lines[from_which:], data_dict[f'{cfg_name}'][eval_tag])
    return data_dict


def save(data_dict):
    save_file_raw = Path(sys.argv[0]).with_suffix('.pkl').name
    save_file = Path("experiments/results/comparison") / save_file_raw
    cnt = 1
    while True:
        if save_file.exists():
            save_file = save_file.parent / (save_file.stem + f"_{cnt}.pkl")
            cnt += 1
        else:
            break
    with open(save_file.__str__(), 'wb') as f:
        print(save_file)
        pickle.dump(data_dict, f)


# def visualization(data_dict, metric_ind=1, metric_type='oo3d', metric_level=4):
def visualization(data_dict, metric_ind=1, metric_type='3d', metric_level=1):
    print(f"--------------------------------\n"
          f"visualization ...")
    fig = plt.figure(figsize=(6.4 * 2.46, 4.8 * 1.9))  # 1382,894
    ax = plt.axes()
    # ax.set_facecolor('#E9E9F1')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.ylabel('AP$_{R40}$@0.7 (car Mod.)', fontsize=26)
    plt.xlabel('slope of the terrain($\circ$)', fontsize=26)
    plt.tick_params(labelsize=25)
    # plt.ylim([30, 100])
    plt.ylim([0.3, 1.0])
    lines = []
    names = []
    results = []
    slope_rates = [name.split(sep='_')[-2] for name in list(data_dict[list(data_dict.keys())[0]].keys())]
    orders = [1, 0, 3, 2, 4, 5] + [6, 7, 8, 9, 10]
    #
    # colors = ['#ff0000', '#ff0069', '#f800d3', '#bb00ff', '#6b00ff', '#0000ff', ] + \
    #          ['#0095ff', '#00f3f3', '#00ff9f', '#00ff00', '#000000']
    # colors = ['#730522', '#2F67C6', '#D24D4A', '#F38B76', '#F9E6DE', '#F6E6DF'] + \
    #          ['#C6E0E8','#86C4DB','#3996C2','#0570AF','#004A7A']
    # colors = ['#AE042C', '#D72930', '#EF5B45', '#FF925D', '#FFC684', '#FFEBAA'] + \
    #          ['#D3F0F4', '#A0D5E3', '#6CB1D0', '#4380B5', '#3256A1']
    colors = ['#AE042C', '#D72930', '#EF5B45', '#FF925D', '#FFC684', '#FFEBAA'] + \
             ['#86C4DB', '#3996C2', '#0570AF', '#004A7A', '#002A9A']
    name_map = {'second': 'SECOND', 'pointpillar': 'PointPillars', 'PartA2_free': 'Part$A^2$',
                'pv_rcnn': 'PV-RCNN', 'centerpoint_nms': 'CenterPoint',
                'voxel_rcnn_car': 'Voxel R-CNN', 'pointrcnn': 'PointRCNN',
                '3dssd_car': '3DSSD', '3dssd_sasa_car': '3DSSD-SASA', 'IA-SSD': 'IA-SSD',
                'det6d_car': 'Det6D (ours)'}
    linewidths = [6] * 10 + [8]
    for midx, (m, ds) in enumerate(data_dict.items()):
        ds_result = []
        for d, p in ds.items():
            metric_info = list(p.keys())[metric_ind]
            metric_result = p[metric_info][metric_type][metric_level]
            ds_result.append(metric_result / 100.0)
        print(f"{m} = {ds_result}")
        names.append(m)
        results.append(ds_result)
        lines += plt.plot(slope_rates, ds_result, labooel=name_map[m], alpha=0.9,
                          linestyle='solid', marker='o', linewidth=linewidths[midx], color=colors[orders[midx]])
    results = np.array(results)
    print(results.shape)
    # dash_line_data = (np.mean(results[0:6, :-1], axis=0) + np.mean(results[6:-1, :-1], axis=0)) / 2
    # plt.plot(slope_rates[:-1], dash_line_data, '--k')

    plt.legend(handles=lines[0:6], labels=[name_map[m] for m in names[0:6]],
               loc='upper left',
               ncol=3,
               frameon=True,
               title="%-15s" % "voxel based",
               fontsize=20,
               title_fontsize=25)
    second_legend = Legend(handles=lines[6:],
                           labels=[name_map[m] for m in names[6:]],
                           ncol=2,
                           loc='upper right',
                           frameon=True,
                           parent=ax,
                           title="%-15s" % "point based",
                           fontsize=20,
                           title_fontsize=25)
    ax.add_artist(second_legend)
    plt.grid()
    plt.tight_layout()
    plt.savefig('experiments/results/performance_comparison.pdf')
    plt.show()
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    cfgs = args.cfgs
    ckpts = args.ckpts
    datasets = args.datasets
    eval_tags = ['eval_' + Path(path).name.__str__() for path in datasets] if datasets is not None else datasets
    output_root = Path("../output/slopedkitti_models")
    print(f"====== collect results ======\n"
          f"output root: {output_root.absolute()}")
    print("models:")
    for cfg, ckpt in zip(cfgs, ckpts):
        print(f"  {cfg} <-> {ckpt}")
    print("datasets:")
    for dataset, eval_tag in zip(datasets, eval_tags):
        print(f"  {dataset} <-> {eval_tag}")

    assert output_root.exists()
    eval_results = None
    if cfgs is None or ckpts is None or datasets is None:
        if args.file is not None:
            print(f'read from file: {args.file}')
            with open(args.file, 'rb') as f:
                eval_results = pickle.load(f)
    else:
        eval_results = analysis(output_root)
        if args.save:
            save(eval_results)

    visualization(eval_results)
