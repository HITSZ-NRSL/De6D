import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25, -1]
    MAX_OCCLUSION = [0, 1, 2, 10000]
    MAX_TRUNCATION = [0.15, 0.3, 0.5, 10000]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):  # 每个gt
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def d9_box_matching_score(boxes, query_boxes, score_type=0):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    match_scores = np.zeros((N, K), dtype=boxes.dtype)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for k in range(K):
        for n in range(N):
            dist = np.linalg.norm(boxes[n][:3] - query_boxes[k][:3])
            diag_b = np.linalg.norm(boxes[n][3:6])
            diag_qb = np.linalg.norm(query_boxes[k][3:6])
            if score_type == 0:  # 2-2xsigmoid(dist)
                match_scores[n, k] = 2 - 2 * sigmoid(dist)
            elif score_type == 1:  # 2-2xsigmoid(dist) * (2xdiag_axdiag_b)/(diag_a**2+diag_b**2)
                match_scores[n, k] = 2 - 2 * sigmoid(dist) * (2 * diag_b * diag_qb) / (diag_qb ** 2 + diag_b ** 2)
            elif score_type == 2:  # 1-2xdist/(diag_a+diag_b)
                match_scores[n, k] = max(1 - 2 * dist / (diag_b + diag_qb), 0)
            # elif score_type == 3:  # 1-2xdist/(diag_a+diag_b)
            #     # for dist score
            #     dist_score = max(1 - 2 * dist / (diag_b + diag_qb), 0)
            #     # for scale score
            #     dim_b, dim_qb = boxes[n][3:6], query_boxes[k][3:6]
            #     intersection = np.min(np.array([dim_b, dim_qb]), axis=0).prod()
            #     scale_score = max(1 - intersection / (dim_b.prod() + dim_qb.prod() - intersection), 0)
            #     # for rotation score
            #     rot_b, rot_q = boxes[n][6:9], query_boxes[k][6:9]
            #     rot_score = max(1 - (np.abs(rot_b - rot_q) / np.array([2 * np.pi, np.pi, np.pi])).sum(), 0)
            #     match_scores[n, k] = dist_score * 3 / 4.0 + (scale_score / 8.0 + rot_score / 8.0) * dist_score > 0
            #     print(match_scores[n, k])
            else:
                raise NotImplementedError
    return match_scores


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size  # 本gt是否被det匹配上
    ignored_threshold = [False] * det_size
    gt_of_tp_detection = np.ones((det_size,), dtype=np.int32) * -1
    thresholds = np.zeros((gt_size,))
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    # 当前gt是否检测到
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False
        # 本gt和所有det进行分析
        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            gt_of_tp_detection[det_idx] = i
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1

    return tp, fp, fn, similarity, thresholds[:thresh_idx], gt_of_tp_detection


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    # 每个场景
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                                                           gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)

            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)

            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 3:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            yaw = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)[..., None]
            pitch = np.concatenate([a["pitch"] for a in gt_annos_part], 0)[..., None]
            roll = np.concatenate([a["roll"] for a in gt_annos_part], 0)[..., None]
            gt_boxes = np.concatenate([loc, dims, yaw, pitch, roll], axis=1)

            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            yaw = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)[..., None]
            pitch = np.concatenate([a["pitch"] for a in dt_annos_part], 0)[..., None]
            roll = np.concatenate([a["roll"] for a in dt_annos_part], 0)[..., None]
            dt_boxes = np.concatenate([loc, dims, yaw, pitch, roll], axis=1)
            overlap_part = d9_box_matching_score(gt_boxes, dt_boxes, score_type=0).astype(np.float64)
            pass
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num, dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):  # 每个场景
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        # 合法gt个数，每个gt是否抛弃的标记，每个det是否抛弃的标记，DontCare的bbox2
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)



def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    # 每个类别，每个难度下，每个iou阈值，每个召回率采样点 —— 上的精度
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    ate = np.zeros([num_class, num_difficulty, num_minoverlap])
    ase = np.zeros([num_class, num_difficulty, num_minoverlap])
    aoe = np.zeros([num_class, num_difficulty, num_minoverlap, 3])
    num_tp = np.zeros([num_class, num_difficulty, num_minoverlap])
    num_tp_sloped = np.zeros([num_class, num_difficulty, num_minoverlap])

    for m, current_class in enumerate(current_classes):  # 某个类别
        # print(f"class: {current_class}")
        for l, difficulty in enumerate(difficultys):  # 某个难度
            # print(f"difficultys: {l}")
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            # gt(bbox,alpha) dt(bbox,alpha,score) 忽略的gt 忽略的检测
            gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt = rets
            if metric == 3 and l == 3:
                print(f"num_dt: {np.array([(scene == 0).sum() for scene in ignored_dets]).sum()}"
                      f", num_dt_ignore_dfc: {np.array([(scene == 1).sum() for scene in ignored_dets]).sum()}"
                      f", num_dt_ignore_cls: {np.array([(scene == -1).sum() for scene in ignored_dets]).sum()}"
                      f", num_gt: {np.array([(scene == 0).sum() for scene in ignored_gts]).sum()}"
                      f", num_gt_ignore_dfc: {np.array([(scene == 1).sum() for scene in ignored_gts]).sum()}"
                      f", num_gt_ignore_cls: {np.array([(scene == -1).sum() for scene in ignored_gts]).sum()}")
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):  # 某个iou阈值下
                thresholdss, gt_of_tpss = [], []
                for i in range(len(gt_annos)):  # 某帧场景
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds, gt_of_tp = rets
                    gt_of_tpss.append(gt_of_tp)
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
                if metric == 3:  # ctr mAP and tp scores
                    # 每个场景的gt_of_tps
                    for scene, gt_idx in enumerate(gt_of_tpss):
                        valid_gt_idx_mask = gt_idx > -1
                        if valid_gt_idx_mask.sum() > 0:
                            valid_gt_idx = gt_idx[valid_gt_idx_mask]
                            dim_gt = gt_annos[scene]["dimensions"][valid_gt_idx, :]
                            loc_gt = gt_annos[scene]["location"][valid_gt_idx, :]
                            rot_gt = np.concatenate([gt_annos[scene]["rotation_y"][valid_gt_idx][..., None],
                                                     gt_annos[scene]["pitch"][valid_gt_idx][..., None],
                                                     gt_annos[scene]["roll"][valid_gt_idx][..., None]
                                                     ], axis=-1) % (np.pi * 2)
                            dim_dt = dt_annos[scene]["dimensions"][valid_gt_idx_mask, :]
                            loc_dt = dt_annos[scene]["location"][valid_gt_idx_mask, :]
                            rot_dt = np.concatenate([dt_annos[scene]["rotation_y"][valid_gt_idx_mask][..., None],
                                                     dt_annos[scene]["pitch"][valid_gt_idx_mask][..., None],
                                                     dt_annos[scene]["roll"][valid_gt_idx_mask][..., None]
                                                     ], axis=-1) % (np.pi * 2)

                            translation_error = np.linalg.norm(loc_gt - loc_dt, axis=-1).sum()

                            intersection = np.min(np.array([dim_gt, dim_dt]), axis=0).prod(axis=1)
                            union = dim_dt.prod(axis=1) + dim_gt.prod(axis=1) - intersection
                            scale_error = (1 - intersection / union).sum()

                            rot_weights = np.array([1.0, 1.0, 1.0])
                            rot_dis = np.abs(rot_dt - rot_gt)
                            ros_dis_mask = rot_dis > np.pi
                            rot_dis[ros_dis_mask] = 2 * np.pi - rot_dis[ros_dis_mask]
                            rot_error = (rot_dis * rot_weights).sum(axis=0)
                            ate[m, l, k] += translation_error
                            ase[m, l, k] += scale_error
                            aoe[m, l, k] += rot_error
                            num_tp[m, l, k] += valid_gt_idx_mask.sum()
                            num_tp_sloped[m, l, k] += (np.abs(rot_gt[:, 1:2]) > np.deg2rad(0.5)).sum()
    print(f"num_tp_sloped:{num_tp_sloped}")
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
        "ate": ate,
        "ase": ase,
        "aoe": aoe,
        "num_tp": num_tp
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def get_tp_score(ate, ase, aoe, num_tp):
    print(f"num_tp:{num_tp}")
    print(f"ate:{ate}")
    print(f"ase:{ase}")
    print(f"aoe:{aoe}")
    aoe = aoe.sum(axis=-1)
    ats = np.clip(1 - ate / num_tp, a_min=0, a_max=1)
    ass = np.clip(1 - ase / num_tp, a_min=0, a_max=1)
    aos = np.clip(1 - aoe / num_tp, a_min=0, a_max=1)
    print(f"ats:{ats}")
    print(f"ass:{ass}")
    print(f"aos:{aos}")
    return np.array([ats, ass, aos])


def get_ods(mAP, tp_score_list):
    num_score_type = tp_score_list.shape[0]
    weight = 1 / (num_score_type * 2)

    print(f"mAP:{mAP / 100}")
    return mAP / 100 / 2.0 + (weight * tp_score_list).sum(axis=0)


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval_slopedkitti(gt_annos,
                        dt_annos,
                        current_classes,
                        min_overlaps,
                        compute_aos=False,
                        PR_detail_dict=None):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2, 3]
    # bbox2d
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']
    # bev 2dbbox, precision[num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])  # mAP[[num_class, num_difficulty, num_minoverlap]
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']
    # bev 3dbbox
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']

    # full pose bbox3d
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 3,
                     min_overlaps)
    mAP_3dctr = get_mAP(ret["precision"])
    mAP_3dctr_R40 = get_mAP_R40(ret["precision"])
    tp_score_list = get_tp_score(ret["ate"], ret["ase"], ret["aoe"], ret["num_tp"])
    nds = get_ods(mAP_3dctr, tp_score_list)
    nds_R40 = get_ods(mAP_3dctr_R40, tp_score_list)
    if PR_detail_dict is not None:
        PR_detail_dict['3dctr'] = ret['precision']
    print(f"num_tp(min_overlap): {ret['num_tp'][:, 3, :]}")
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_3dctr, nds, \
           mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40, mAP_3dctr_R40, nds_R40, tp_score_list


def get_slopedkitti_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    # car pedestrain cyclist van person_setting truck
    overlap_0_7 = np.array([[0.70, 0.50, 0.50, 0.70, 0.50, 0.70],  # imgbbox2d / iou
                            [0.70, 0.50, 0.50, 0.70, 0.50, 0.70],  # bev rbbox2d / iou
                            [0.70, 0.50, 0.50, 0.70, 0.50, 0.70],  # bev rbbox3d / iou
                            [0.53, 0.53, 0.53, 0.53, 0.53, 0.53]  # box3d center / meter: 2-2xsimoid(1)
                            ])
    overlap_0_5 = np.array([[0.70, 0.50, 0.50, 0.70, 0.50, 0.50],  # imgbbox2d / iou
                            [0.50, 0.25, 0.25, 0.50, 0.25, 0.50],  # bev rbbox2d / iou
                            [0.50, 0.25, 0.25, 0.50, 0.25, 0.50],  # bev rbbox3d / iou
                            [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],  # box3d center / meter
                            ])
    # min_overlaps[num_overlap, metric, cls].
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]  # min_overlaps[num_overlap, metric, class].
    result = '\n'
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    # calculate metric
    res = do_eval_slopedkitti(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos,
                              PR_detail_dict=PR_detail_dict)

    mAPbbox, mAPbev, mAP3d, mAPaos, mAP3dctr, nds, \
    mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40, mAP3dctr_R40, nds_R40, tp_scores = res

    # print and save result
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            # for 11 recall position
            result += print_str(f"{class_to_name[curcls]} "
                                "AP@{:.2f}, {:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j]))
            result += print_str(f"level  :  easy     mode    hard      all")
            result += print_str(f"bbox AP:"
                                f"{mAPbbox[j, 0, i]:.4f}, "
                                f"{mAPbbox[j, 1, i]:.4f}, "
                                f"{mAPbbox[j, 2, i]:.4f}")
            result += print_str(f"bev  AP:"
                                f"{mAPbev[j, 0, i]:.4f}, "
                                f"{mAPbev[j, 1, i]:.4f}, "
                                f"{mAPbev[j, 2, i]:.4f}")
            result += print_str(f"3d   AP:"
                                f"{mAP3d[j, 0, i]:.4f}, "
                                f"{mAP3d[j, 1, i]:.4f}, "
                                f"{mAP3d[j, 2, i]:.4f}")
            if compute_aos:
                result += print_str((f"aos  AP:"
                                     f"{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(f"3d  CAP:                           {mAP3dctr[j, 3, i]:.4f}")
            result += print_str(f"3d  ATS:                           {tp_scores[0][j, 3, i]:.4f}")
            result += print_str(f"3d  ASS:                           {tp_scores[1][j, 3, i]:.4f}")
            result += print_str(f"3d  AOS:                           {tp_scores[2][j, 3, i]:.4f}")
            result += print_str(f"3d  ODS:                           {nds[j, 3, i]:.4f}")

            result += print_str(f" ")

            # for 40 recall position
            result += print_str(f"{class_to_name[curcls]} "
                                "AP_R40@{:.2f}, {:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j]))
            result += print_str(f"level  :  easy     mode    hard      all")
            result += print_str(f"bbox AP:"
                                f"{mAPbbox_R40[j, 0, i]:.4f}, "
                                f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                f"{mAPbbox_R40[j, 2, i]:.4f}")
            result += print_str(f"bev  AP:"
                                f"{mAPbev_R40[j, 0, i]:.4f}, "
                                f"{mAPbev_R40[j, 1, i]:.4f}, "
                                f"{mAPbev_R40[j, 2, i]:.4f}")
            result += print_str(f"3d   AP:"
                                f"{mAP3d_R40[j, 0, i]:.4f}, "
                                f"{mAP3d_R40[j, 1, i]:.4f}, "
                                f"{mAP3d_R40[j, 2, i]:.4f}")
            if compute_aos:
                result += print_str(f"aos  AP:"
                                    f"{mAPaos_R40[j, 0, i]:.2f}, "
                                    f"{mAPaos_R40[j, 1, i]:.2f}, "
                                    f"{mAPaos_R40[j, 2, i]:.2f}")
                if i == 0:
                    ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                    ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                    ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            result += print_str(f"3d  CAP:                           {mAP3dctr_R40[j, 3, i]:.4f}")
            result += print_str(f"3d  ATS:                           {tp_scores[0][j, 3, i]:.4f}")
            result += print_str(f"3d  ASS:                           {tp_scores[1][j, 3, i]:.4f}")
            result += print_str(f"3d  AOS:                           {tp_scores[2][j, 3, i]:.4f}")
            result += print_str(f"3d  ODS:                           {nds_R40[j, 3, i]:.4f}")

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

            result += print_str(f" ")
    return result, ret_dict
