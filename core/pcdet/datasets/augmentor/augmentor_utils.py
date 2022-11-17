import numpy as np
import math
import copy
from ...utils import common_utils
from ...utils import box_utils
from scipy.spatial.transform import Rotation


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)

        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes


def random_translation_along_x(gt_boxes, points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 0] += offset
    gt_boxes[:, 0] += offset

    # if gt_boxes.shape[1] > 7:
    #     gt_boxes[:, 7] += offset

    return gt_boxes, points


def random_translation_along_y(gt_boxes, points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 1] += offset
    gt_boxes[:, 1] += offset

    # if gt_boxes.shape[1] > 8:
    #     gt_boxes[:, 8] += offset

    return gt_boxes, points


def random_translation_along_z(gt_boxes, points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 2] += offset
    gt_boxes[:, 2] += offset

    return gt_boxes, points


def random_local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 0] += offset

        gt_boxes[idx, 0] += offset

        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[idx, 7] += offset

    return gt_boxes, points


def random_local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 1] += offset

        gt_boxes[idx, 1] += offset

        # if gt_boxes.shape[1] > 8:
        #     gt_boxes[idx, 8] += offset

    return gt_boxes, points


def random_local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 2] += offset

        gt_boxes[idx, 2] += offset

    return gt_boxes, points


def global_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    # threshold = max - length * uniform(0 ~ 0.2)
    threshold = np.max(points[:, 2]) - intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))

    points = points[points[:, 2] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] < threshold]
    return gt_boxes, points


def global_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])

    threshold = np.min(points[:, 2]) + intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    points = points[points[:, 2] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] > threshold]

    return gt_boxes, points


def global_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])

    threshold = np.max(points[:, 1]) - intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] < threshold]

    return gt_boxes, points


def global_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])

    threshold = np.min(points[:, 1]) + intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] > threshold]

    return gt_boxes, points


def local_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points

    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # augs[f'object_{idx}'] = noise_scale
        points_in_box, mask = get_points_in_box(points, box)

        # tranlation to axis center
        points[mask, 0] -= box[0]
        points[mask, 1] -= box[1]
        points[mask, 2] -= box[2]

        # apply scaling
        points[mask, :3] *= noise_scale

        # tranlation back to original position
        points[mask, 0] += box[0]
        points[mask, 1] += box[1]
        points[mask, 2] += box[2]

        gt_boxes[idx, 3:6] *= noise_scale
    return gt_boxes, points


def local_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # augs[f'object_{idx}'] = noise_rotation
        points_in_box, mask = get_points_in_box(points, box)

        centroid_x = box[0]
        centroid_y = box[1]
        centroid_z = box[2]

        # tranlation to axis center
        points[mask, 0] -= centroid_x
        points[mask, 1] -= centroid_y
        points[mask, 2] -= centroid_z
        box[0] -= centroid_x
        box[1] -= centroid_y
        box[2] -= centroid_z

        # apply rotation
        points[mask, :] = common_utils.rotate_points_along_z(points[np.newaxis, mask, :], np.array([noise_rotation]))[0]
        box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][
            0]

        # tranlation back to original position
        points[mask, 0] += centroid_x
        points[mask, 1] += centroid_y
        points[mask, 2] += centroid_z
        box[0] += centroid_x
        box[1] += centroid_y
        box[2] += centroid_z

        gt_boxes[idx, 6] += noise_rotation
        if gt_boxes.shape[1] > 8:
            gt_boxes[idx, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[idx, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]

    return gt_boxes, points


def local_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]

        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z + dz / 2) - intensity * dz

        points = points[np.logical_not(np.logical_and(mask, points[:, 2] >= threshold))]

    return gt_boxes, points


def local_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]

        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z - dz / 2) + intensity * dz

        points = points[np.logical_not(np.logical_and(mask, points[:, 2] <= threshold))]

    return gt_boxes, points


def local_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]

        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y + dy / 2) - intensity * dy

        points = points[np.logical_not(np.logical_and(mask, points[:, 1] >= threshold))]

    return gt_boxes, points


def local_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]

        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y - dy / 2) + intensity * dy

        points = points[np.logical_not(np.logical_and(mask, points[:, 1] <= threshold))]

    return gt_boxes, points


def get_points_in_box(points, gt_box):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz

    MARGIN = 1e-1
    cosa, sina = math.cos(-rz), math.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa

    mask = np.logical_and(abs(shift_z) <= dz / 2.0,
                          np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN,
                                         abs(local_y) <= dy / 2.0 + MARGIN))

    points = points[mask]

    return points, mask


def local_pyramid_dropout(gt_boxes, points, dropout_prob):
    boxes_need_to_drop_mask = np.random.uniform(0, 1, gt_boxes.shape[0]) <= dropout_prob
    if boxes_need_to_drop_mask.sum() != 0:
        box_pyramids = box_utils.boxes_to_pyramids(gt_boxes[boxes_need_to_drop_mask])  # (N,6,15)
        each_box, random_one_face = np.arange(box_pyramids.shape[0]), np.random.randint(0, 6, box_pyramids.shape[0])
        pyramids_in_box_to_drop = box_pyramids[each_box, random_one_face]
        point_to_drop_masks = box_utils.points_in_pyramids_mask(points, pyramids_in_box_to_drop)
        points = points[np.logical_not(point_to_drop_masks.any(-1))]
    return gt_boxes, points


def local_pyramid_sparsify(gt_boxes, points, prob, max_num_pts):
    boxes_need_sparse_mask = np.random.uniform(0, 1, gt_boxes.shape[0]) <= prob
    if boxes_need_sparse_mask.sum() != 0:
        box_pyramids = box_utils.boxes_to_pyramids(gt_boxes[boxes_need_sparse_mask])  # (N,6,15)
        each_box, random_one_face = np.arange(box_pyramids.shape[0]), np.random.randint(0, 6, box_pyramids.shape[0])
        pyramids_in_each_box = box_pyramids[each_box, random_one_face]
        point_in_each_pyramids_masks = box_utils.points_in_pyramids_mask(points, pyramids_in_each_box)
        num_pts_in_each_pyramids = point_in_each_pyramids_masks.sum(0)
        valid_pyramids_mask = num_pts_in_each_pyramids > max_num_pts
        if np.sum(valid_pyramids_mask) != 0:
            valid_pyramids = pyramids_in_each_box[valid_pyramids_mask]
            point_in_each_valid_pyramids_masks = point_in_each_pyramids_masks[:, valid_pyramids_mask]
            remain_points = points[np.logical_not(point_in_each_valid_pyramids_masks.any(-1))]

            filtered_points = np.zeros([valid_pyramids.shape[0] * max_num_pts, points.shape[1]])
            for i in range(valid_pyramids.shape[0]):
                indices = np.random.choice(point_in_each_valid_pyramids_masks[:, i].sum(), size=max_num_pts)
                filtered_points[i * max_num_pts:(i + 1) * max_num_pts, :] = \
                    points[point_in_each_valid_pyramids_masks[:, i]][indices]

            points = np.concatenate([remain_points, filtered_points], axis=0)
    return gt_boxes, points


def local_pyramid_swap(gt_boxes, points, swap_prob, max_num_pts):
    def get_points_ratio(pts, pyramid):
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[
                                                                                                 0:3] - surface_center
        alphas = ((pts[:, 0:3] - pyramid[3:6]) * vector_0).sum(-1) / np.power(vector_0, 2).sum()
        betas = ((pts[:, 0:3] - pyramid[3:6]) * vector_1).sum(-1) / np.power(vector_1, 2).sum()
        gammas = ((pts[:, 0:3] - surface_center) * vector_2).sum(-1) / np.power(vector_2, 2).sum()
        return [alphas, betas, gammas]

    def recover_points_by_ratio(points_ratio, pyramid):
        alphas, betas, gammas = points_ratio
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[
                                                                                                 0:3] - surface_center
        points = (alphas[:, None] * vector_0 + betas[:, None] * vector_1) + pyramid[3:6] + gammas[:, None] * vector_2
        return points

    def recover_points_intensity_by_ratio(points_intensity_ratio, max_intensity, min_intensity):
        return points_intensity_ratio * (max_intensity - min_intensity) + min_intensity

    boxes_need_swap_mask = np.random.uniform(0, 1, (gt_boxes.shape[0])) <= swap_prob
    if boxes_need_swap_mask.sum() > 0:
        pyramids_in_boxes_need_swap = box_utils.boxes_to_pyramids(gt_boxes[boxes_need_swap_mask]).reshape([-1, 6, 15])
        point_pyramids_masks = box_utils.points_in_pyramids_mask(points, pyramids_in_boxes_need_swap)
        point_nums = point_pyramids_masks.sum(0).reshape(pyramids_in_boxes_need_swap.shape[0], -1)
        # ignore dropout pyramids or highly occluded pyramids, selected boxes and all their valid pyramids
        valid_boxes_pyramid_mask = np.array(point_nums > max_num_pts)  # bool(N,6)
        if np.sum(valid_boxes_pyramid_mask).sum() > 0:
            # select source box and their corresponding pyramid
            valid_boxes_index, valid_pyramid_in_box_index = np.nonzero(valid_boxes_pyramid_mask)
            swap_source_box_indices = np.unique(valid_boxes_index)
            swap_source_pyramid_indices = [np.random.choice(valid_pyramid_in_box_index[valid_boxes_index == i]) \
                                           for i in swap_source_box_indices]

            # using source pyramid(the same with target pyramid) to select target box
            valid_boxes_index, valid_pyramid_in_box_index = swap_source_box_indices, swap_source_pyramid_indices
            swap_target_box_indices = np.array([np.random.choice(np.where(valid_boxes_pyramid_mask[:, j])[0]) \
                                                    if np.where(valid_boxes_pyramid_mask[:, j])[0].shape[0] > 0 else i
                                                for i, j in zip(valid_boxes_index, valid_pyramid_in_box_index)])
            swap_source_pyramid_indices = np.array(swap_source_pyramid_indices)

            # detected points which in the source&target pyramid
            swap_not_in_same_box = np.logical_not(swap_source_box_indices == swap_target_box_indices)
            if np.sum(swap_not_in_same_box) != 0:
                swap_source_box_indices = swap_source_box_indices[swap_not_in_same_box]
                swap_target_box_indices = swap_target_box_indices[swap_not_in_same_box]
                swap_source_pyramid_indices = swap_source_pyramid_indices[swap_not_in_same_box]
                swap_source_pyramids = pyramids_in_boxes_need_swap[swap_source_box_indices, swap_source_pyramid_indices]
                swap_target_pyramids = pyramids_in_boxes_need_swap[swap_target_box_indices, swap_source_pyramid_indices]
                swap_pyramids_pairs = np.concatenate([swap_source_pyramids, swap_target_pyramids], axis=0)
                swap_point_masks = box_utils.points_in_pyramids_mask(points, swap_pyramids_pairs)
                remain_points = points[np.logical_not(swap_point_masks.any(-1))]
                # print(swap_source_box_indices, swap_target_box_indices, swap_source_pyramid_indices)

                points_res = []
                num_target_pyramids = swap_target_pyramids.shape[0]
                for i in range(num_target_pyramids):
                    source_pyramid = swap_source_pyramids[i]
                    target_pyramid = swap_target_pyramids[i]

                    src_pts = points[swap_point_masks[:, i]]
                    tag_pts = points[swap_point_masks[:, i + num_target_pyramids]]

                    # for intensity transform
                    src_pts_intensity_ratio = (src_pts[:, -1:] - src_pts[:, -1:].min()) \
                                              / np.clip((src_pts[:, -1:].max() - src_pts[:, -1:].min()), 1e-6, 1)
                    tag_pts_intensity_ratio = (tag_pts[:, -1:] - tag_pts[:, -1:].min()) \
                                              / np.clip((tag_pts[:, -1:].max() - tag_pts[:, -1:].min()), 1e-6, 1)

                    src_pts_ratio = get_points_ratio(src_pts, source_pyramid.reshape(15))
                    tag_pts_ratio = get_points_ratio(tag_pts, target_pyramid.reshape(15))
                    new_src_pts = recover_points_by_ratio(tag_pts_ratio, source_pyramid.reshape(15))
                    new_tag_pts = recover_points_by_ratio(src_pts_ratio, target_pyramid.reshape(15))

                    # for intensity transform
                    new_src_pts_intensity = recover_points_intensity_by_ratio(
                        tag_pts_intensity_ratio, src_pts[:, -1:].max(), src_pts[:, -1:].min())
                    new_tag_pts_intensity = recover_points_intensity_by_ratio(
                        src_pts_intensity_ratio, tag_pts[:, -1:].max(), tag_pts[:, -1:].min())

                    new_src_pts = np.concatenate([new_src_pts, new_src_pts_intensity], axis=1)
                    new_tag_pts = np.concatenate([new_tag_pts, new_tag_pts_intensity], axis=1)

                    points_res.append(new_src_pts)
                    points_res.append(new_tag_pts)
                points_res = np.concatenate(points_res, axis=0)
                points = np.concatenate([remain_points, points_res], axis=0)
    return gt_boxes, points


def random_global_make_slope(gt_boxes, points, params=None, rotate_point=None, rotate_angle=None, smooth=False):
    def random(n=1):
        """ make uniform distribution in [-1,1] """
        return (np.random.random(n) - 0.5) * 2

    # get rotate point and rotate angle
    assert params is not None
    dist_mean, dist_var, angle_mean, angle_var = params
    if rotate_point is None:
        mean, var = np.array([dist_mean, 0]), np.array([dist_var, 0])
        polar_pos = mean + random(2) * var
        rotate_point = np.array([polar_pos[0] * np.cos(polar_pos[1]), polar_pos[0] * np.sin(polar_pos[1]), 0])

    x0, y0 = rotate_point[0], rotate_point[1]
    if rotate_angle is None:
        mean, var = angle_mean, angle_var
        k0 = y0 / x0
        k1 = -1 / (k0 + 1e-6)
        v = np.array([x0 - 0, y0 - (-x0 * k1 + y0), 0])
        v /= np.linalg.norm(v)
        angle = mean + random() * var
        v *= angle
        direction = np.sign(np.cross(rotate_point, v)[2])
        # v *= -1 if direction > 0 else 1
        rotate_angle = v

    # apply sloped-aug in smooth way (resolution)
    if smooth:
        temp_rotate_point = rotate_point
        temp_rotate_angle = rotate_angle
        radius, bins = rotate_point[0] / np.abs(rotate_angle[1]), 2
        alpha = rotate_angle[1]
        dist = rotate_point[0]
        for theta in np.linspace(0, alpha, bins):
            delta = alpha / bins
            center = np.array([dist, 0, radius])
            rotate_point = center + np.array([-radius * np.sin(theta), 0, -radius * np.cos(theta)])
            rotate_angle = np.array([0, delta, 0])
            # rotated this block of slope by given params
            gt_boxes, points, rotate_point, rotate_angle = random_global_make_slope(
                gt_boxes, points,
                params=(20, 10, *np.deg2rad([20, 8])),
                rotate_angle=rotate_angle,
                rotate_point=rotate_point)
        return gt_boxes, points, temp_rotate_point, temp_rotate_angle
    else:
        k = rotate_angle[1] / (rotate_angle[0] + 1e-6)
        sign = np.sign(k * (0 - x0) + y0 - 0)
        in_plane_mask = np.sign(k * (points[:, 0] - x0) + y0 - points[:, 1]) != sign
        slope_points = points[in_plane_mask]
        slope_points[:, 0:3] -= rotate_point
        rot = Rotation.from_rotvec(rotate_angle).as_matrix()
        slope_points[:, 0:3] = (slope_points[:, 0:3].dot(rot.T))
        slope_points[:, 0:3] += rotate_point
        points[in_plane_mask] = slope_points

        if gt_boxes.shape[1] < 9:
            gt_boxes = np.concatenate((gt_boxes, np.zeros([gt_boxes.shape[0], 2])), axis=1)
        in_plane_mask = np.sign(k * (gt_boxes[:, 0] - x0) + y0 - gt_boxes[:, 1]) != sign
        slope_box = gt_boxes[in_plane_mask]
        slope_box[:, :3] -= rotate_point
        slope_box[:, :3] = (slope_box[:, :3].dot(rot.T))
        slope_box[:, :3] += rotate_point
        gt_boxes[in_plane_mask] = slope_box
        euler = Rotation.from_rotvec(rotate_angle).as_euler('XYZ')

        # boxes(9)[x, y, z, dx, dy, dz, rz(, ry, rx)[Rot=RxRyRz]]
        gt_boxes[in_plane_mask, 7] += euler[1]  # y
        gt_boxes[in_plane_mask, 8] += euler[0]  # y
        gt_boxes[:, 6:9] = common_utils.limit_period(
            gt_boxes[:, 6:9], offset=0.5, period=2 * np.pi
        )
        return gt_boxes, points, rotate_point, rotate_angle
