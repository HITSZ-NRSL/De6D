import time

import torch
import torch.nn as nn
import math
from torch.autograd import Function, Variable

from . import pointnet2_stack_cuda as pointnet2

import numba
import numpy as np


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (
            str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx,
                                                  new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply


@numba.jit(nopython=True, cache=True)
def pca_svd(x):
    n, c = x.shape
    mean_x = x[:, 0].mean()
    mean_y = x[:, 1].mean()
    mean_z = x[:, 2].mean()
    mean = np.array([mean_x, mean_y, mean_z])
    x_centeried = x - mean
    u, s, v = np.linalg.svd(x_centeried)
    components = v[:c].T
    return s, components, mean_z


@numba.jit('(float32[:,:],int64[:,:,:], int32[:, :])', nopython=True, cache=True)
def analysis(points, patchs_point_idx, patchs_point_count):
    ground_valid_map = np.zeros_like(patchs_point_count, dtype=np.int32)  # ground 1, non_ground 0, unknown -1
    ground_height_map = np.zeros_like(patchs_point_count, dtype=np.float32)
    ground_normal = np.zeros((*patchs_point_count.shape, 3), dtype=np.float32)

    # coarse seg
    for i in range(patchs_point_count.shape[0]):
        for j in range(patchs_point_count.shape[1]):
            count = patchs_point_count[i, j]
            if count > 2:
                points_idx = patchs_point_idx[i, j, :count]
                patch = points[points_idx, :3]
                s, v, mean_z = pca_svd(patch)
                if ((np.linalg.norm(np.cross(v[..., 0], [0, 0, 1])) > 0.8
                     and np.linalg.norm(np.cross(v[..., 1], [0, 0, 1])) > 0.8)
                        and s[1] / (s[2] + 1e-6) > 5):
                    ground_height_map[i, j] = mean_z
                    ground_valid_map[i, j] = 1
                    ground_normal[i, j] = v[..., 2]

    return ground_normal, ground_height_map, ground_valid_map


@numba.jit('(int32[:, :], int32[:, :], int64[:, :, :])', nopython=True, cache=True)
def generate_patchs_kernel(points_grid_idx, grid_count, point_patchs):
    max_num_point = point_patchs.shape[2]
    for i in range(points_grid_idx.shape[0]):
        grid_x, grid_y = points_grid_idx[i, 0], points_grid_idx[i, 1]
        cnt = grid_count[grid_x, grid_y]
        if cnt < max_num_point:
            point_patchs[grid_x, grid_y, cnt] = i
            grid_count[grid_x, grid_y] += 1


def generate_patch(points, voxel_size):
    point_grid_idx = np.round(points[:, 0:2] / np.array(voxel_size)).astype(np.int32)

    min_grid_idx = np.min(point_grid_idx, axis=0)
    max_grid_idx = np.max(point_grid_idx, axis=0)
    point_grid_idx = point_grid_idx - min_grid_idx

    grid_size = max_grid_idx - min_grid_idx + 1
    grid_count = np.zeros(grid_size, dtype=np.int32)
    point_idx_patchs = np.zeros((*grid_size, 50), dtype=np.int64)

    generate_patchs_kernel(point_grid_idx, grid_count, point_idx_patchs)

    return point_grid_idx, point_idx_patchs, grid_count, min_grid_idx, grid_size


def ground_aware_farthest_point_sampling(points: torch.Tensor, num_sampled_points: int) -> torch.Tensor:
    def ground_estimation(points, voxel_size=[2, 2]):
        point_idx, patchs_point_idx, patch_point_counts, min_grid, grid_size = generate_patch(points,
                                                                                              voxel_size=voxel_size)
        normal_dense, height_dense, valid_dense = analysis(points, patchs_point_idx, patch_point_counts)
        non_ground = np.arange(point_idx.shape[0])[valid_dense[point_idx[:, 0], point_idx[:, 1]] == 0]
        return non_ground

    cpu_points = points.cpu().numpy()  # [B,N,C]
    sample_idx = []
    non_ground_idx = []
    xyz_points_list, xyz_batch_cnt_list, num_sampled_points_list = [], [], []
    num_batch = points.shape[0]

    for b in range(num_batch):
        cur_cpu_points = cpu_points[b, ...]  # (N,C)
        non_ground_idx.append(torch.from_numpy(ground_estimation(cur_cpu_points)).to(points.device))
        xyz_points_list.append(points[b, non_ground_idx[b], :])
        xyz_batch_cnt_list.append(len(non_ground_idx[b]))
        num_sampled_points_list.append(num_sampled_points)

    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt_list, device=points.device).int()
    new_xyz_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()
    sampled_pt_idxs = stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, new_xyz_batch_cnt
    ).long()

    idx_offset = 0
    slice_start = 0
    for b in range(num_batch):
        slice_end = int((b + 1) * num_sampled_points)
        sample_idx.append(non_ground_idx[b][sampled_pt_idxs[slice_start:slice_end] - idx_offset])

        slice_start += int(num_sampled_points)
        idx_offset += len(non_ground_idx[b])

    sampled_idx = torch.cat(sample_idx, dim=0).contiguous().view(num_batch, num_sampled_points).int()
    return sampled_idx


def gd_farthest_point_sampling(points: torch.Tensor, num_sampled_points: int) -> torch.Tensor:
    def ground_estimation(points, voxel_size=[2, 2]):
        point_idx, patchs_point_idx, patch_point_counts, min_grid, grid_size = \
            generate_patch(points, voxel_size=voxel_size)
        normal_dense, height_dense, valid_dense = analysis(points, patchs_point_idx, patch_point_counts)
        return point_idx, patch_point_counts, valid_dense

    cpu_points = points.cpu().numpy()  # [B,N,C]
    sample_idx = []
    num_batch = points.shape[0]

    for b in range(num_batch):
        cur_cpu_points = cpu_points[b, ...]  # (N,C)
        grid_idx, count_grid, ground_grid = ground_estimation(cur_cpu_points)
        grid_idx_ = (grid_idx[:, 0], grid_idx[:, 1])
        point_fg_flag = 1 - ground_grid[grid_idx_]
        point_dense = count_grid[grid_idx_]
        point_sampled_prob = (53 - point_dense) * (point_fg_flag + 1)
        probabilities = point_sampled_prob / point_sampled_prob.sum()
        sampled_pt_idxs = np.random.choice(cur_cpu_points.shape[0], size=num_sampled_points, replace=False,
                                           p=probabilities)
        sample_idx.append(torch.from_numpy(sampled_pt_idxs).to(points.device))
    sampled_idx = torch.cat(sample_idx, dim=0).contiguous().view(num_batch, num_sampled_points).int()
    return sampled_idx


def dense_aware_point_sampling(points: torch.Tensor, num_sampled_points: int) -> torch.Tensor:
    # num_bs, num_pts, num_fet = points.shape
    #
    # POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 30]
    # VOXEL_SIZE = [2, 2, POINT_CLOUD_RANGE[-1] - POINT_CLOUD_RANGE[-1 - 3]]
    # point_cloud_range = torch.Tensor(POINT_CLOUD_RANGE).cuda().float()
    # voxel_size = torch.Tensor(VOXEL_SIZE).cuda().float()
    # grid_size = torch.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).cuda().long()
    # scale_xy = grid_size[0] * grid_size[1]
    # scale_y = grid_size[1]
    #
    # point_batch_idx = torch.arange(num_bs, device=points.device)[..., None].repeat(1, num_pts).view(-1)
    # point_pillar_coor = torch.floor((points[..., :2] - point_cloud_range[:2]) / voxel_size[:2]).long().view(-1, 2)
    # voxel_coor_id = point_batch_idx.int() * scale_xy + point_pillar_coor[:, 0] * scale_y + point_pillar_coor[:, 1]
    # unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
    # point_dense_estimated = unq_count[unq_inv].view(num_bs, -1)
    # point_sampled_prob = (1 / point_dense_estimated).cpu().numpy()
    #
    # sample_idx = []
    # for b in range(num_bs):
    #     sample_prob = point_sampled_prob[b]
    #     sample_prob = sample_prob / sample_prob.sum()
    #     sampled_pt_idxs = np.random.choice(num_pts, size=num_sampled_points, replace=False, p=sample_prob)
    #     sample_idx.append(torch.from_numpy(sampled_pt_idxs).to(points.device))
    # sampled_idx = torch.cat(sample_idx, dim=0).contiguous().view(num_bs, num_sampled_points).int()
    POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 30]
    VOXEL_SIZE = [0.1, 2 * np.pi]

    num_bs, num_pts, num_fet = points.shape
    point_cloud_range = points.new_tensor(POINT_CLOUD_RANGE)
    cylinder_range = points.new_tensor([0, 0, torch.norm(point_cloud_range[3:5] - point_cloud_range[0:2]), 2 * np.pi])
    cylinder_voxel_size = points.new_tensor(VOXEL_SIZE)
    cylinder_grid_size = torch.round((cylinder_range[2:4] - cylinder_range[0:2]) / cylinder_voxel_size).long()
    scale_rho_phi = cylinder_grid_size[0] * cylinder_grid_size[1]
    scale_phi_ = cylinder_grid_size[1]

    rho = torch.norm(points[..., 0:2], dim=-1)  # xoy
    phi = torch.atan2(points[..., 1], points[..., 0]) + np.pi / 2
    cylinder_coords = torch.cat((rho[..., None], phi[..., None]), dim=-1)
    cylinder_coords = torch.floor(cylinder_coords / cylinder_voxel_size).long().view(-1, 2)
    point_batch_idx = torch.arange(num_bs, device=points.device)[..., None].repeat(1, num_pts).view(-1)
    voxel_coor_id = point_batch_idx.int() * scale_rho_phi + cylinder_coords[:, 0] * scale_phi_ + cylinder_coords[:, 1]
    unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
    point_dense_estimated = unq_count[unq_inv].view(num_bs, -1)
    point_sampled_prob = (1 / point_dense_estimated).cpu().numpy()

    sample_idx = []
    for b in range(num_bs):
        sample_prob = point_sampled_prob[b]
        sample_prob = sample_prob / sample_prob.sum()
        sampled_pt_idxs = np.random.choice(num_pts, size=num_sampled_points, replace=False, p=sample_prob)
        sample_idx.append(torch.from_numpy(sampled_pt_idxs).to(points.device))
    sampled_idx = torch.cat(sample_idx, dim=0).contiguous().view(num_bs, num_sampled_points).int()
    return sampled_idx


def dense_aware_point_sampling_single(points: torch.Tensor, num_sampled_points: int) -> torch.Tensor:
    POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 30]
    VOXEL_SIZE = [0.1, 2 * np.pi]

    num_pts, num_fet = points.shape
    point_cloud_range = points.new_tensor(POINT_CLOUD_RANGE)
    cylinder_range = points.new_tensor([0, 0, torch.norm(point_cloud_range[3:5] - point_cloud_range[0:2]), 2 * np.pi])
    cylinder_voxel_size = points.new_tensor(VOXEL_SIZE)
    cylinder_grid_size = torch.round((cylinder_range[2:4] - cylinder_range[0:2]) / cylinder_voxel_size).long()
    scale_phi_ = cylinder_grid_size[1]

    rho = torch.norm(points[..., 0:2], dim=-1)  # xoy
    phi = torch.atan2(points[..., 1], points[..., 0]) + np.pi / 2
    cylinder_coords = torch.cat((rho[..., None], phi[..., None]), dim=-1)
    cylinder_coords = torch.floor(cylinder_coords / cylinder_voxel_size).long().view(-1, 2)
    voxel_coor_id = cylinder_coords[:, 0] * scale_phi_ + cylinder_coords[:, 1]
    unq_coor, unq_inv, unq_count = torch.unique(voxel_coor_id, return_inverse=True, return_counts=True, dim=0)
    point_dense_estimated = unq_count[unq_inv]
    point_sampled_prob = (1 / point_dense_estimated)
    point_sampled_prob = point_sampled_prob / point_sampled_prob.sum()
    point_sampled_prob = point_sampled_prob.cpu().numpy()
    sampled_pt_idxs = np.random.choice(num_pts, size=num_sampled_points,
                                       replace=True if num_pts < num_sampled_points else False, p=point_sampled_prob)
    return sampled_pt_idxs


def sectorized_farthest_point_sampling(points: torch.Tensor, num_sampled_points: int, num_sectors: int) -> torch.Tensor:
    """
    Args:
        points: (B, N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_idx: (B, N_out)
    """
    num_batch = points.shape[0]
    sector_size = math.pi * 2 / num_sectors
    sampled_idx = []
    for b in range(num_batch):
        cur_points = points[b, ...]
        xyz_points_list, xyz_batch_cnt_list, num_sampled_points_list = [], [], []
        point_angles = torch.atan2(cur_points[:, 1], cur_points[:, 0]) + math.pi
        sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)  # (B,N)
        for k in range(num_sectors):
            sector_mask = (sector_idx == k)
            cur_num_points = sector_mask.sum().item()
            if cur_num_points > 0:
                xyz_points_list.append(cur_points[sector_mask])
                xyz_batch_cnt_list.append(cur_num_points)
                ratio = cur_num_points / cur_points.shape[0]
                num_sampled_points_list.append(min(cur_num_points, math.ceil(ratio * num_sampled_points)))

        if len(xyz_batch_cnt_list) == 0:
            xyz_points_list.append(cur_points)
            xyz_batch_cnt_list.append(len(cur_points))
            num_sampled_points_list.append(num_sampled_points)
            print(f'Warning: empty sector points detected in SectorFPS: points.shape={cur_points.shape}')

        xyz = torch.cat(xyz_points_list, dim=0)
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt_list, device=cur_points.device).int()
        new_xyz_batch_cnt = torch.tensor(num_sampled_points_list, device=cur_points.device).int()

        sampled_pt_idxs = stack_farthest_point_sample(
            xyz.contiguous(), xyz_batch_cnt, new_xyz_batch_cnt
        ).int()

        sampled_idx.append(sampled_pt_idxs[:num_sampled_points])
    sampled_idx = torch.cat(sampled_idx, dim=0).contiguous().view(num_batch, num_sampled_points)
    return sampled_idx


class StackFarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, xyz_batch_cnt, npoint):
        """
        Args:
            ctx:
            xyz: (N1 + N2 + ..., 3) where N > npoint
            xyz_batch_cnt: [N1, N2, ...]
            npoint: int, number of features in the sampled set

        Returns:
            output: (npoint.sum()) tensor containing the set,
            npoint: (M1, M2, ...)
        """
        assert xyz.is_contiguous() and xyz.shape[1] == 3

        batch_size = xyz_batch_cnt.__len__()
        if not isinstance(npoint, torch.Tensor):
            if not isinstance(npoint, list):
                npoint = [npoint for i in range(batch_size)]
            npoint = torch.tensor(npoint, device=xyz.device).int()

        N, _ = xyz.size()
        temp = torch.cuda.FloatTensor(N).fill_(1e10)
        output = torch.cuda.IntTensor(npoint.sum().item())

        pointnet2.stack_farthest_point_sampling_wrapper(xyz, temp, xyz_batch_cnt, output, npoint)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


stack_farthest_point_sample = StackFarthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, unknown_batch_cnt, known, known_batch_cnt):
        """
        Args:
            ctx:
            unknown: (N1 + N2..., 3)
            unknown_batch_cnt: (batch_size), [N1, N2, ...]
            known: (M1 + M2..., 3)
            known_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
            idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
        """
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = unknown.new_zeros(unknown.shape)
        idx = unknown_batch_cnt.new_zeros(unknown.shape).int()

        pointnet2.three_nn_wrapper(
            unknown.contiguous(), unknown_batch_cnt.contiguous(),
            known.contiguous(), known_batch_cnt.contiguous(), dist2, idx
        )
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        pointnet2.three_interpolate_wrapper(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(
            grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class ThreeNNForVectorPoolByTwoStep(Function):
    @staticmethod
    def forward(ctx, support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                max_neighbour_distance, nsample, neighbor_type, avg_length_of_neighbor_idxs, num_total_grids,
                neighbor_distance_multiplier):
        """
        Args:
            ctx:
            // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            // xyz_batch_cnt: (batch_size), [N1, N2, ...]
            // new_xyz: (M1 + M2 ..., 3) centers of the ball query
            // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            // nsample: find all (-1), find limited number(>0)
            // neighbor_type: 1: ball, others: cube
            // neighbor_distance_multiplier: query_distance = neighbor_distance_multiplier * max_neighbour_distance

        Returns:
            // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
            // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
        """
        num_new_xyz = new_xyz.shape[0]
        new_xyz_grid_dist2 = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape)
        new_xyz_grid_idxs = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape).int().fill_(-1)

        while True:
            num_max_sum_points = avg_length_of_neighbor_idxs * num_new_xyz
            stack_neighbor_idxs = new_xyz_grid_idxs.new_zeros(num_max_sum_points)
            start_len = new_xyz_grid_idxs.new_zeros(num_new_xyz, 2).int()
            cumsum = new_xyz_grid_idxs.new_zeros(1)

            pointnet2.query_stacked_local_neighbor_idxs_wrapper_stack(
                support_xyz.contiguous(), xyz_batch_cnt.contiguous(),
                new_xyz.contiguous(), new_xyz_batch_cnt.contiguous(),
                stack_neighbor_idxs.contiguous(), start_len.contiguous(), cumsum,
                avg_length_of_neighbor_idxs, max_neighbour_distance * neighbor_distance_multiplier,
                nsample, neighbor_type
            )
            avg_length_of_neighbor_idxs = cumsum[0].item() // num_new_xyz + int(cumsum[0].item() % num_new_xyz > 0)

            if cumsum[0] <= num_max_sum_points:
                break

        stack_neighbor_idxs = stack_neighbor_idxs[:cumsum[0]]
        pointnet2.query_three_nn_by_stacked_local_idxs_wrapper_stack(
            support_xyz, new_xyz, new_xyz_grid_centers, new_xyz_grid_idxs, new_xyz_grid_dist2,
            stack_neighbor_idxs, start_len, num_new_xyz, num_total_grids
        )

        return torch.sqrt(new_xyz_grid_dist2), new_xyz_grid_idxs, torch.tensor(avg_length_of_neighbor_idxs)


three_nn_for_vector_pool_by_two_step = ThreeNNForVectorPoolByTwoStep.apply


class VectorPoolWithVoxelQuery(Function):
    @staticmethod
    def forward(ctx, support_xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, support_features: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor, num_grid_x, num_grid_y, num_grid_z,
                max_neighbour_distance, num_c_out_each_grid, use_xyz,
                num_mean_points_per_grid=100, nsample=-1, neighbor_type=0, pooling_type=0):
        """
        Args:
            ctx:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            support_features: (N1 + N2 ..., C)
            new_xyz: (M1 + M2 ..., 3) centers of new positions
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            num_grid_x: number of grids in each local area centered at new_xyz
            num_grid_y:
            num_grid_z:
            max_neighbour_distance:
            num_c_out_each_grid:
            use_xyz:
            neighbor_type: 1: ball, others: cube:
            pooling_type: 0: avg_pool, 1: random choice
        Returns:
            new_features: (M1 + M2 ..., num_c_out)
        """
        assert support_xyz.is_contiguous()
        assert support_features.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        num_total_grids = num_grid_x * num_grid_y * num_grid_z
        num_c_out = num_c_out_each_grid * num_total_grids
        N, num_c_in = support_features.shape
        M = new_xyz.shape[0]

        assert num_c_in % num_c_out_each_grid == 0, \
            f'the input channels ({num_c_in}) should be an integral multiple of num_c_out_each_grid({num_c_out_each_grid})'

        while True:
            new_features = support_features.new_zeros((M, num_c_out))
            new_local_xyz = support_features.new_zeros((M, 3 * num_total_grids))
            point_cnt_of_grid = xyz_batch_cnt.new_zeros((M, num_total_grids))

            num_max_sum_points = num_mean_points_per_grid * M
            grouped_idxs = xyz_batch_cnt.new_zeros((num_max_sum_points, 3))

            num_cum_sum = pointnet2.vector_pool_wrapper(
                support_xyz, xyz_batch_cnt, support_features, new_xyz, new_xyz_batch_cnt,
                new_features, new_local_xyz, point_cnt_of_grid, grouped_idxs,
                num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance, use_xyz,
                num_max_sum_points, nsample, neighbor_type, pooling_type
            )
            num_mean_points_per_grid = num_cum_sum // M + int(num_cum_sum % M > 0)
            if num_cum_sum <= num_max_sum_points:
                break

        grouped_idxs = grouped_idxs[:num_cum_sum]

        normalizer = torch.clamp_min(point_cnt_of_grid[:, :, None].float(), min=1e-6)
        new_features = (new_features.view(-1, num_total_grids, num_c_out_each_grid) / normalizer).view(-1, num_c_out)

        if use_xyz:
            new_local_xyz = (new_local_xyz.view(-1, num_total_grids, 3) / normalizer).view(-1, num_total_grids * 3)

        num_mean_points_per_grid = torch.Tensor([num_mean_points_per_grid]).int()
        nsample = torch.Tensor([nsample]).int()
        ctx.vector_pool_for_backward = (point_cnt_of_grid, grouped_idxs, N, num_c_in)
        ctx.mark_non_differentiable(new_local_xyz, num_mean_points_per_grid, nsample, point_cnt_of_grid)
        return new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid

    @staticmethod
    def backward(ctx, grad_new_features: torch.Tensor, grad_local_xyz: torch.Tensor, grad_num_cum_sum,
                 grad_point_cnt_of_grid):
        """
        Args:
            ctx:
            grad_new_features: (M1 + M2 ..., num_c_out), num_c_out = num_c_out_each_grid * num_total_grids

        Returns:
            grad_support_features: (N1 + N2 ..., C_in)
        """
        point_cnt_of_grid, grouped_idxs, N, num_c_in = ctx.vector_pool_for_backward
        grad_support_features = grad_new_features.new_zeros((N, num_c_in))

        pointnet2.vector_pool_grad_wrapper(
            grad_new_features.contiguous(), point_cnt_of_grid, grouped_idxs,
            grad_support_features
        )

        return None, None, grad_support_features, None, None, None, None, None, None, None, None, None, None, None, None


vector_pool_with_voxel_query_op = VectorPoolWithVoxelQuery.apply

if __name__ == '__main__':
    pass
