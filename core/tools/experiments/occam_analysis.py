import argparse
import pickle
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.utils.analysis.occam import OccAM

import open3d
from tools.visual_utils import open3d_vis_utils as V


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str,
                        default='cfgs/kitti_models/pointpillar.yaml',
                        help='dataset/model config for the demo')
    parser.add_argument('--occam_cfg_file', type=str,
                        default='cfgs/occam_configs/kitti.yaml',
                        help='specify the OccAM config')
    parser.add_argument('--source_file_path', type=str, default='demo_pcl.npy',
                        help='point cloud data file to analyze')
    parser.add_argument('--ckpt', type=str, default=None, required=True,
                        help='path to pretrained model parameters')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for OccAM creation')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--nr_it', type=int, default=6000,
                        help='number of sub-sampling iterations N')
    parser.add_argument('--viz', action='store_true', default=False)

    args = parser.parse_args()

    cfg_from_yaml_file(args.model_cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg


def main():
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM Demo -------------------------')
    save_path = Path('experiments/results/occam')
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / f"{Path(args.source_file_path).stem}_{args.nr_it}.pkl"

    occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                  occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

    if save_path.exists():
        logger.info(f'Read from cache: {save_path}')
        logger.info('Visualize attribution map of first object')
        with open(save_path, 'rb') as f:
            info_dict = pickle.load(f)
        for i in range(info_dict['boxes'].shape[0]):
            # attr: conf, trans, scale, rx(,ry,rz)
            occam.visualize_attr_map(info_dict['points'],
                                     info_dict['boxes'][i, :],
                                     info_dict['attr'][i, :, 4])
        return

    pcl = occam.load_and_preprocess_pcl(args.source_file_path)

    # get detections to analyze (in full pcl)
    base_det = occam.get_base_predictions(pcl=pcl)
    base_det_boxes, base_det_labels, base_det_scores = base_det

    logger.info('Number of detected objects to analyze: '
                + str(base_det_labels.shape[0]))

    logger.info('Start attribution map computation:')

    attr_maps = occam.compute_attribution_maps(
        pcl=pcl, base_det_boxes=base_det_boxes,
        base_det_labels=base_det_labels, batch_size=args.batch_size,
        num_workers=args.workers)

    logger.info('DONE')
    if args.viz:
        logger.info('Visualize attribution map of first object')
        occam.visualize_attr_map(pcl, base_det_boxes[0, :], attr_maps.prod(axis=-1)[0, :])
    with open(save_path, 'wb') as f:
        logger.info(f'save to file: {save_path}')
        pickle.dump({'points': pcl, 'attr': attr_maps, 'boxes': base_det_boxes}, f)


if __name__ == '__main__':
    main()
