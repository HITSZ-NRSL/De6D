import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ..model_utils.pointmlp_utils import index_points, knn_point, square_distance, _gather
from ...utils.common_utils import TimeMeasurement


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


# @对输入点进行局部group(knn)，并对局部点进行normalize（geometry affine），有助于学习稳定
class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param use_xyz: 输出特征是否附加group采样中心的xyz坐标
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0  # @ 使用xyz，输出维度附加xyz作为额外特征
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        """

        Parameters
        ----------
        xyz: xyz坐标
        points: 点云+特征？

        Returns
        -------

        """
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        # @ 输入点云采样group个数作为group中心点
        with TimeMeasurement("PointMLP::Encoder::Group::FPS") as timex:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = _gather(xyz, fps_idx)  # [B, npoint, 3]
        new_points = _gather(points, fps_idx)  # [B, npoint, d]
        with TimeMeasurement("PointMLP::Encoder::Group::KNN") as timex:
            idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = _gather(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = _gather(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)  # @dim=2：对group内k个点求均值
            elif self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            else:
                raise NotImplementedError
            # (grouped_points - mean).reshape(B, -1) -> (B,npoint*k*d+3)
            # std -> (B,1=δ,1,1)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            # @ alpha相当于给每个通道加权
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        # @(B, npoint, k, d)+(B, npoint, k, d)
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        # @调整通道
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        # @batch_size npoints neighbor fea_channle
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        # @conv_nd输入要求为(b,c,*size_nd)
        x = x.reshape(-1, d, s)  # @(b*npoint, fea_channel, neighbor)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        # @ aggregation 对邻域内进行channel-wise max pool
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        with TimeMeasurement("PointMLP::Decoder::FP::interpolate") as timex:
            points2 = points2.permute(0, 2, 1)
            B, N, C = xyz1.shape
            _, S, _ = xyz2.shape

            if S == 1:
                interpolated_points = points2.repeat(1, N, 1)
            else:
                dists = square_distance(xyz1, xyz2)  # @[B, N, M]
                # with TimeMeasurement("PointMLP::Decoder::FP::KNN3") as timex:
                #     dists, idx = dists.sort(dim=-1)  # @与xyz1最近的xyz2
                #     dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 每个xyz1取距离他最近的xyz2
                with TimeMeasurement("PointMLP::Decoder::FP::KNN3") as timex:
                    dists, idx = torch.topk(dists, 3, dim=-1, largest=False, sorted=False)
                dist_recip = 1.0 / (dists + 1e-8)  # @[B, N, 3]
                norm = torch.sum(dist_recip, dim=2, keepdim=True)  # @[B, N, 1]
                weight = dist_recip / norm  # @[B, N, 3]
                # @从points2中按idx找到3对应的3个feature，按xyz1到它们的距离倒数加权
                interpolated_points = torch.sum(_gather(points2, idx) * weight.view(B, N, 3, 1), dim=2)

            if points1 is not None:
                points1 = points1.permute(0, 2, 1)
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = interpolated_points
            new_points = new_points.permute(0, 2, 1)

        with TimeMeasurement("PointMLP::Decoder::FP::Fusion") as timex:
            new_points = self.fuse(new_points)

        with TimeMeasurement("PointMLP::Decoder::FP::Extraction") as timex:
            new_points = self.extraction(new_points)
        return new_points


class PointMLPBackBone(nn.Module):
    """
    REFERENCE: https://arxiv.org/abs/2202.07123
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        """
        Parameters
        ----------
        num_classes: 分类数
        points：输入点数
        embed_dim：输入点云嵌入到高维后输入网路
        groups：卷积输入输出通道是否分组进行
        res_expansion：res-block输出输入维度比值
        activation：激活函数
        bias：卷积层是否有偏置
        use_xyz：采样group后特征是否拼接邻域点的xyz
        normalize：group内标准化方法
        dim_expansion：encoder每个stage增加输出维度的倍数
        pre_blocks：每个stage中φ_pre内有多少res block(隐含一个输入通道调整block:ConvBNReLU1D，固2个block其实是有3个)
        pos_blocks：每个stage中φ_pos内有多少res block(隐含一个输入通道调整block，固2个block其实是有3个)
        k_neighbors：knn邻域个数
        reducers：每stage点云数量缩减倍数
        de_dims：decoder每层输出维度
        de_blocks：decoder每层中的res-block
        gmp_dim：每层编码层输出结果经过gmp提取到gmp_dim通道的特征，然后用于max_pooling得到本层编码的全局上下文。
        cls_dim: 类别信息one-hot输入经过(num_cls,cls_dim,cls_dim)得到一个类型特征(cls_token)
        kwargs
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.bias, self.use_xyz = True, True
        self.activation, self.normalize = 'relu', 'anchor'
        self.stages = len(model_cfg.ENCODER.NPOINTS)

        """ build embedding """
        self.embedding = ConvBNReLU1D(3, self.model_cfg.EMBED, bias=self.bias, activation=self.activation)

        """ build encoder """
        assert len(model_cfg.ENCODER.NPOINTS) \
               == len(model_cfg.ENCODER.POS_EXTRACTION.RES_BLOCK) \
               == len(model_cfg.ENCODER.PRE_EXTRACTION.MLPS) \
               == len(model_cfg.ENCODER.PRE_EXTRACTION.RES_BLOCK) \
               == len(model_cfg.ENCODER.PRE_EXTRACTION.KNN)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        input_channel = self.model_cfg.EMBED
        encoder_dims = [input_channel]

        for i in range(len(model_cfg.ENCODER.NPOINTS)):
            out_channel = model_cfg.ENCODER.PRE_EXTRACTION.MLPS[i]

            # local_grouper_list
            # input: x(B,N,3) feat(B,N,C)
            # return: new_xyz(B,M,3), new_feat(B,M,K,2xC(+3)) 其中2xC(+3): C(group_feat)+3(group_xyz)+C(new_xyz.reapeat)
            self.local_grouper_list.append(
                LocalGrouper(channel=input_channel,
                             groups=model_cfg.ENCODER.NPOINTS[i],
                             kneighbors=model_cfg.ENCODER.PRE_EXTRACTION.KNN[i],
                             use_xyz=self.use_xyz, normalize=self.normalize)
            )
            # pre_block_list
            # input: feat(B,M,K,2xC(+3))
            # return: new_feat(B,out_channel,M)
            self.pre_blocks_list.append(
                PreExtraction(channels=input_channel,
                              out_channels=out_channel,
                              blocks=model_cfg.ENCODER.PRE_EXTRACTION.RES_BLOCK[i],
                              groups=1, res_expansion=1, activation=self.activation,
                              bias=self.bias, use_xyz=self.use_xyz)
            )

            # append pos_block_list
            # input: feat(B,out_channel,M)
            # return: new_feat(B,out_channel*1(res_expansion),M)
            self.pos_blocks_list.append(
                PosExtraction(channels=out_channel,
                              blocks=model_cfg.ENCODER.POS_EXTRACTION.RES_BLOCK[i],
                              groups=1, res_expansion=1,
                              activation=self.activation, bias=self.bias)
            )

            input_channel = out_channel
            encoder_dims.append(input_channel)

        """ build decoder """
        self.decode_list = nn.ModuleList()
        encoder_dims.reverse()  # (512, 256, 128, 64, 32)
        decoder_dims = copy.deepcopy(model_cfg.DECODER.MLPS)
        decoder_dims.insert(0, encoder_dims[0])  # (512, 512, 256, 128, 128)
        assert len(encoder_dims) == len(decoder_dims) == len(model_cfg.DECODER.MLPS) + 1

        for i in range(len(encoder_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(
                    in_channel=decoder_dims[i] + encoder_dims[i + 1],
                    out_channel=decoder_dims[i + 1],
                    blocks=model_cfg.DECODER.RES_BLOCK[i],
                    groups=1, res_expansion=1,
                    bias=self.bias, activation=self.activation)
            )

        # self.act = get_activation(self.activation)
        #
        # # class label mapping
        # self.cls_map = nn.Sequential(
        #     ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
        #     ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        # )
        #
        # # global max pooling mapping
        # self.gmp_map_list = nn.ModuleList()
        # for en_dim in encoder_dims:
        #     self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        # self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(encoder_dims), gmp_dim, bias=bias, activation=activation)

        # classifier
        # self.classifier = nn.Sequential(
        #     nn.Conv1d(gmp_dim + cls_dim + de_dims[-1], 128, 1, bias=bias),
        #     nn.BatchNorm1d(128),
        #     self.act,
        #     nn.Dropout(),
        #     nn.Conv1d(128, num_classes, 1, bias=bias)
        # )
        self.num_point_features = self.model_cfg.DECODER.MLPS[-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_features: (num_points, C)
        """
        with TimeMeasurement("PointMLP") as time1:
            batch_size = batch_dict['batch_size']
            points = batch_dict['points']
            batch_idx, xyz, features = self.break_up_pc(points)

            xyz = xyz.view(batch_size, -1, 3)  # xyz(B,N,3)
            feat = xyz.permute(0, 2, 1)  # feat(B,3,N)

            """ feed to network """

            with TimeMeasurement("PointMLP::Embedding") as time2:
                feat = self.embedding(feat)  # (B, 3, N) -> (B, 64, N)

            xyz_list = [xyz]  # [B, N, 3]
            feat_list = [feat]  # [B, D, N]

            # here is the encoder
            # operation: φ_pos(A(φ_pre(f_ij)))
            with TimeMeasurement("PointMLP::Encoder") as time3:
                for i in range(self.stages):
                    # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
                    with TimeMeasurement("PointMLP::Encoder::Group{}".format(i)) as timex:
                        xyz, feat = self.local_grouper_list[i](xyz, feat.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
                    with TimeMeasurement("PointMLP::Encoder::PreExtraction".format(i)) as timex:
                        feat = self.pre_blocks_list[i](feat)  # [b,d,g]
                    with TimeMeasurement("PointMLP::Encoder::PosExtraction".format(i)) as timex:
                        feat = self.pos_blocks_list[i](feat)  # [b,d,g]
                    xyz_list.append(xyz)
                    feat_list.append(feat)

            # here is the decoder
            xyz_list.reverse()
            feat_list.reverse()
            feat = feat_list[0]

            with TimeMeasurement("PointMLP::Decoder") as time4:
                for i in range(len(self.decode_list)):
                    with TimeMeasurement("PointMLP::Decoder::FP{}".format(i)) as timex:
                        feat = self.decode_list[i](xyz_list[i + 1], xyz_list[i], feat_list[i + 1], feat)

            point_features = feat.permute(0, 2, 1).contiguous()  # (B, N, C)
            batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
            batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), xyz_list[-1].view(-1, 3)), dim=1)
            return batch_dict
