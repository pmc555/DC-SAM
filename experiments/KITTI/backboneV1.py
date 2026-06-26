import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.layers import VNLinear, VNLinearLeakyReLU, VNLeakyReLU, VNStdFeature
from pareconv.modules.ops import index_select


# ---------------- Channel Aggregation (CA) 模块 ----------------
class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)),
                                  requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """Channel Reallocation / Channel Aggregation Feed-Forward Network"""
    def __init__(self, embed_dims, ffn_ratio=4., kernel_size=3, ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        feedforward_channels = int(embed_dims * ffn_ratio)

        # 1x1 conv -> depthwise conv -> 1x1 conv
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, 1)
        self.dwconv = nn.Conv2d(feedforward_channels, feedforward_channels,
                                kernel_size, 1, kernel_size // 2,
                                groups=feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, 1)
        self.drop = nn.Dropout(ffn_drop)

        # 特征分解增强
        self.decompose = nn.Conv2d(feedforward_channels, 1, 1)
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5)
        self.decompose_act = nn.GELU()

        # 新增归一化层
        self.norm = nn.BatchNorm2d(embed_dims)

    def feat_decompose(self, x):
        temp = self.decompose(x)
        temp = self.decompose_act(temp)
        temp = x - temp
        temp = self.sigma(temp)
        return x + temp

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        # 残差 + 归一化
        x = x + residual
        x = self.norm(x)
        return x


# ---------------- 改进后的 CorrelationNet ----------------
class CorrelationNet(nn.Module):
    """
    CA-Enhanced CorrelationNet
    支持通过通道重分配 (CA Block) 生成动态卷积核权重。
    完全兼容旧版接口（保留 hidden_unit 参数）。
    """
    def __init__(self, in_channel, out_channel,
                 hidden_unit=None,          # 👈 兼容旧版参数
                 last_bn=False, temp=1,
                 ca_ffn_ratio=2.0, ca_kernel_size=3, ca_ffn_drop=0.):
        super(CorrelationNet, self).__init__()

        # --- VN 等变特征提取层 ---
        self.vn_layer = VNLinearLeakyReLU(in_channel, out_channel * 2, dim=4,
                                          share_nonlinearity=False, negative_slope=0.2)
        self.temp = temp

        # --- Channel Aggregation Block ---
        self.ca_block = ChannelAggregationFFN(
            embed_dims=out_channel * 2,
            ffn_ratio=ca_ffn_ratio,
            kernel_size=ca_kernel_size,
            ffn_drop=ca_ffn_drop
        )

        # --- 降维 + BN ---
        self.reduce = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel) if last_bn else nn.Identity()

    def forward(self, xyz):
        """
        xyz : [N, D, 3, K]
        return : [N, out_channel, K]
        """
        N, _, _, K = xyz.size()

        # --- Step 1: 等变特征编码 ---
        scores = self.vn_layer(xyz)                # [N, out_channel*2, 3, K]
        scores = torch.norm(scores, p=2, dim=2)    # [N, out_channel*2, K]

        # --- Step 2: CA 聚合增强 ---
        scores = scores.unsqueeze(-1)              # [N, C, K, 1]
        #residual = scores
        scores = self.ca_block(scores)             # [N, C, K, 1]

        # --- Step 3: 降维 + 残差 + 归一化 ---
        scores = self.reduce(scores)               # [N, out_channel, K, 1]
        scores = self.bn(scores)
        #scores = scores + residual[:, :scores.size(1), :, :]
        scores = scores.squeeze(-1)                # [N, out_channel, K]

        # --- Step 4: softmax 生成邻域权重 ---
        scores = F.softmax(scores / self.temp, dim=1)
        return scores


class PARE_Conv_Block(nn.Module):
    """几何 bootstrap 卷积块（仅用于第一层）。

    刻意不接 conv_info：此层忽略 s_feats，只从 pts/centers/cross 纯几何特征构造初始
    feature（weightbank dim = in_dim+2）。不要"统一"成 PARE_Conv_Resblock 的 edge_conv/
    use_xyz 行为——第一层还没有有意义的输入特征，且改 weightbank 维度会破坏预训练权重。
    """
    def __init__(self, in_dim, out_dim, kernel_size, share_nonlinearity=False):
        super(PARE_Conv_Block, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        in_dim = in_dim + 2   # 1 + 2: [xyz, mean, cross]
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, in_dim, out_dim // 2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(in_dim, self.kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        """
        q_pts N1 * 3
        s_pts N2 * 3
        idx   N1 * k
        s_feats N2 * D * 3
        """
        N, K = neighbor_indices.shape
        # print(f"q_pts shape: {q_pts.shape}")
        # print(f"s_pts shape: {s_pts.shape}")
        # print(f"neighbor_indices shape: {neighbor_indices.shape}")
        # print(f"neighbor_indices min: {neighbor_indices.min().item()}, max: {neighbor_indices.max().item()}")
        # compute relative coordinates
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)  # [N, 1, 3, K]
        centers = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], 1) # [N, 3, 3, K] rotation equivariant spatial features

        # predict correlation scores
        scores = self.score_net(local_feats) # [N, kernel_size,  K]

        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', local_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N,  self.kernel_size, -1, 3, K)

        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1) # [N, D/2, 3, K]
        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)
        # mean pooling
        new_feats = normed_feats.mean(-1)
        # applying VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # mapping D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]

        return new_feats

class PARE_Conv_Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, shortcut_linear=False, share_nonlinearity=False, conv_info=None):
        super(PARE_Conv_Resblock, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        self.conv_way = conv_info["conv_way"]
        self.use_xyz = conv_info["use_xyz"]
        conv_dim = in_dim * 2 if self.conv_way == 'edge_conv' else in_dim
        if self.use_xyz: conv_dim += 1
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, conv_dim, out_dim//2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(conv_dim, self.kernel_size * out_dim//2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.shortcut_proj = VNLinear(in_dim, out_dim) if shortcut_linear else nn.Identity()
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)
    def forward(self, q_pts, s_pts, s_feats, idx):
        """
        q_pts N1 * 3
        s_pts N2 * 3
        idx   N1 * k
        feats N2 * D * 3
        """

        N, K = idx.shape
        pts = (s_pts[idx] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)    # N1 *1 * 3 * k
        # compute relative coordinates
        center = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, center, dim=2)
        local_feats = torch.cat([pts, center, cross], 1)# [N, 3, 3, K] rotation equivariant spatial features
        # predict correlation scores
        scores = self.score_net(local_feats)
        # gather neighbors features
        neighbor_feats = s_feats[idx, :].permute(0, 2, 3, 1)                            # N1  D * 3 k
        # shortcut
        identify = neighbor_feats[..., 0]
        identify = self.shortcut_proj(identify)
        # get edge features
        if self.conv_way == 'edge_conv':
            q_feats = neighbor_feats[..., 0:1]
            neighbor_feats = torch.cat([neighbor_feats - q_feats, neighbor_feats], 1)
        # use relative coordinates
        if self.use_xyz:
            neighbor_feats = torch.cat([neighbor_feats, pts], 1)
        # use correlation scores to assemble features
        #print(f"neighbor_feats: {neighbor_feats.shape}")
        #print(f"self.weightbank: {self.weightbank}")
        pro_feats = torch.einsum('ncdk,cf->nfdk', neighbor_feats, self.weightbank)

        pro_feats = pro_feats.reshape(N, self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1)


        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)
        # mean pooling
        new_feats = normed_feats.mean(-1)
        # apply VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # map D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]
        # add shortcut
        #print("new_feats shape:", new_feats.shape)
        #print("identify shape:", identify.shape)
        new_feats = new_feats + identify

        return new_feats

class PAREConvFPN(nn.Module):
    def __init__(self, init_dim, output_dim, kernel_size, share_nonlinearity=False, conv_way='edge_conv', use_xyz=True, use_encoder_re_feats=True):
        super(PAREConvFPN, self).__init__()
        conv_info = {'conv_way': conv_way, 'use_xyz': use_xyz}
        self.use_encoder_re_feats = use_encoder_re_feats
        self.encoder2_1 = PARE_Conv_Block(1, init_dim // 3, kernel_size, share_nonlinearity=share_nonlinearity)
        self.encoder2_2 = PARE_Conv_Resblock(init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder2_3 = PARE_Conv_Resblock(2 * init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.encoder3_1 = PARE_Conv_Resblock(2 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_2 = PARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_3 = PARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.encoder4_1 = PARE_Conv_Resblock(4 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_2 = PARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_3 = PARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.coarse_RI_head = VNLinear(8 * init_dim // 3, 8 * init_dim // 3)
        self.coarse_std_feature = VNStdFeature(8 * init_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.decoder3 = VNLinearLeakyReLU(12 * init_dim // 3, 4 * init_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.decoder2 = VNLinearLeakyReLU(6 * init_dim // 3, output_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.RI_head = VNLinear(output_dim // 3, output_dim // 3)
        self.RE_head = VNLinear(output_dim // 3, output_dim // 3)

        self.fine_std_feature = VNStdFeature(output_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.matching_score_proj = nn.Linear(output_dim // 3 * 3, 1)

    def forward(self, data_dict):
        # feats_list = []
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        feats_s1 = points_list[0][:, None]
        # feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        coarse_feats = self.coarse_RI_head(feats_s4)
        RI_feats_c, _ = self.coarse_std_feature(coarse_feats)

        RI_feats_c = RI_feats_c.reshape(RI_feats_c.shape[0], -1)

        up1 = upsampling_list[1]
        latent_s3 = index_select(feats_s4, up1[:, 0], dim=0)
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        up2 = upsampling_list[0]
        latent_s2 = index_select(latent_s3, up2[:, 0], dim=0)
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        ri_feats = self.RI_head(latent_s2)
        re_feats = self.RE_head(latent_s2)

        ri_feats_f, local_rot = self.fine_std_feature(ri_feats)
        ri_feats_f = ri_feats_f.reshape(ri_feats_f.shape[0], -1)
        m_scores = self.matching_score_proj(ri_feats_f).sigmoid().squeeze()
        if not self.training and self.use_encoder_re_feats:
            # using rotation equivariant features from encoder to solve transformation may generate better hypotheses,
            # probably because a larger receptive field would contaminate rotation equivariant features
            re_feats_f = feats_s2
        else:
            re_feats_f = re_feats
        return re_feats_f, ri_feats_f, feats_s4, RI_feats_c, m_scores

