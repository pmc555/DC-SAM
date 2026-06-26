import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance
from pareconv.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
from pareconv.modules.transformer.conditional_transformer import  ExtendedConditionalTransformer

class GeometricStructureEmbedding(nn.Module): #几何结构编码：生成融合距离和角度的几何嵌入
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d # 距离缩放因子
        self.sigma_a = sigma_a  # 角度缩放因子
        self.factor_a = 180.0 / (self.sigma_a * np.pi) # 弧度转角度
        self.angle_k = angle_k  # 近邻数

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim) # 正弦位置编码
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)   # 距离投影层
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)   # 角度投影层

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)

        # ref_vectors = normals.unsqueeze(2)

        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        # a_embeddings = a_embeddings[:, :, :, 0, :]
        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='Relu',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list


class ExtendedGeometricTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            num_heads,
            blocks,
            sigma_d,
            sigma_a,
            angle_k,
            k=2,
            spots=1,  # 新增参数：spot数量
            spot_k=2,  # 新增参数：每个spot包含的点数
            sigma_c=1.8,  # 新增参数：兼容性计算的尺度
            seed_threshold=0.3,  # 新增参数：种子点选择阈值
            seed_num=5,  # 新增参数：种子点数量
            dual_normalization=True,  # 新增参数：是否使用双重归一化
            dropout=None,
            activation_fn='Relu',
            reduction_a='max',
    ):
        super(ExtendedGeometricTransformer, self).__init__()

        # 保存新参数
        self.k = k
        self.spots = spots
        self.spot_k = spot_k
        self.sigma_c = sigma_c
        self.seed_threshold = seed_threshold
        self.seed_num = seed_num
        self.dual_normalization = dual_normalization

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # 替换原始Transformer为支持新模块的版本
        self.transformer = ExtendedConditionalTransformer(
            blocks, hidden_dim, num_heads, spots, spot_k, sigma_c,
            seed_threshold, seed_num, dual_normalization,
            dropout, activation_fn, return_attention_scores=True
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
            self,
            ref_points,
            src_points,
            ref_feats,
            src_feats,
            ref_masks=None,
            src_masks=None,
    ):
        # 计算几何嵌入

        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        # 特征投影
        # print(f"ref_feats 形状: {ref_feats.shape}")
        # print(f"ref_feats 形状: {ref_feats.shape}")
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)



        # 计算距离矩阵
        k = max(self.k + 1, self.spot_k)  # 确保有足够的近邻点
        with torch.no_grad():
            ref_dists = pairwise_distance(ref_points, ref_points)
            src_dists = pairwise_distance(src_points, src_points)

            # 计算K近邻索引

            ref_idx = ref_dists.topk(k, largest=False).indices
            src_idx = src_dists.topk(k, largest=False).indices

        # 扩展Transformer前向传播
        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats, src_feats,
            ref_embeddings, src_embeddings,
            ref_dists, src_dists,
            ref_idx, src_idx,
            masks0=ref_masks,
            masks1=src_masks,
        )

        # 输出投影
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list


class ExtendedGeometricTransformer1(nn.Module):
    def __init__(
            self,
            input_dim1,
            output_dim,
            hidden_dim,
            num_heads,
            blocks1,
            sigma_d,
            sigma_a,
            angle_k,
            k=12,
            spots=4,  # 新增参数：spot数量
            spot_k=12,  # 新增参数：每个spot包含的点数
            sigma_c=1.8,  # 新增参数：兼容性计算的尺度
            seed_threshold=0.3,  # 新增参数：种子点选择阈值
            seed_num=48,  # 新增参数：种子点数量
            dual_normalization=True,  # 新增参数：是否使用双重归一化
            dropout=None,
            activation_fn='ReLU',
            reduction_a='max',
    ):
        super(ExtendedGeometricTransformer1, self).__init__()

        # 保存新参数
        self.k = k
        self.spots = spots
        self.spot_k = spot_k
        self.sigma_c = sigma_c
        self.seed_threshold = seed_threshold
        self.seed_num = seed_num
        self.dual_normalization = dual_normalization

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.in_proj = nn.Linear(input_dim1, hidden_dim)

        # 替换原始Transformer为支持新模块的版本
        self.transformer = ExtendedConditionalTransformer(
            blocks1, hidden_dim, num_heads, spots, spot_k, sigma_c,
            seed_threshold, seed_num, dual_normalization,
            dropout, activation_fn, return_attention_scores=True
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
            self,
            ref_points,
            src_points,
            ref_feats,
            src_feats,
            ref_masks=None,
            src_masks=None,
    ):
        # 计算几何嵌入

        # ref_embeddings = self.embedding(ref_points)
        # src_embeddings = self.embedding(src_points)

        # 特征投影
        # print(f"ref_feats 形状: {ref_feats.shape}")
        # print(f"ref_feats 形状: {ref_feats.shape}")
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)



        # 计算距离矩阵
        k = max(self.k + 1, self.spot_k)  # 确保有足够的近邻点
        with torch.no_grad():
            ref_dists = pairwise_distance(ref_points, ref_points)
            src_dists = pairwise_distance(src_points, src_points)

            # 计算K近邻索引

            ref_idx = ref_dists.topk(k, largest=False).indices
            src_idx = src_dists.topk(k, largest=False).indices

        # 扩展Transformer前向传播
        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats, src_feats,
            ref_points, src_points,
            ref_dists, src_dists,
            ref_idx, src_idx,
            masks0=ref_masks,
            masks1=src_masks,
        )

        # 输出投影
        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list
