import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pareconv.modules.sinkhorn.learnable_sinkhorn import LearnableLogOptimalTransport

class PointDualMatching(nn.Module):
    def __init__(self, dim):
        """point dual matching"""
        super(PointDualMatching, self).__init__()
        self.proj1 = nn.Linear(dim, dim, True)
        self.inf = np.inf

    def forward(self, ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks):
        """point dual matching forward.
        Args:
            ref_node_corr_knn_feats: torch.Tensor (N, k, D)
            src_node_corr_knn_feats: torch.Tensor (N, k, D)
            ref_node_corr_knn_scores: torch.Tensor (N, k)
            src_node_corr_knn_scores: torch.Tensor (N, k)
            ref_node_corr_knn_masks: torch.bool (N, k)
            src_node_corr_knn_masks: torch.bool (N, k)

        Returns:
            matching_scores: torch.Tensor (N, k, k)
        """
        m_ref_feats, m_src_feats = self.proj1(ref_node_corr_knn_feats), self.proj1(src_node_corr_knn_feats)

        scores = torch.einsum('bnd,bmd->bnm', m_ref_feats, m_src_feats)  # (P, K, K)
        scores = scores / m_ref_feats.shape[-1] ** 0.5

        batch_size, num_row, num_col = scores.shape
        device = scores.device
        padded_row_masks = torch.zeros((batch_size, num_row, num_col), device=device)
        padded_row_masks.masked_fill_(~src_node_corr_knn_masks[:, None, :], float('-inf'))

        padded_col_masks = torch.zeros((batch_size, num_row, num_col), device=device)
        padded_col_masks.masked_fill_(~ref_node_corr_knn_masks[:, :, None], float('-inf'))
        matching_scores = F.softmax(scores + padded_row_masks, -1) * F.softmax(scores + padded_col_masks, 1)
        matching_scores = matching_scores * ref_node_corr_knn_scores[:, :, None] * src_node_corr_knn_scores[:, None, :]
        return matching_scores

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class LinearAttention(nn.Module):
    """线性注意力 φ(Q)(φ(K)^T V) / (φ(Q)·Σφ(K)) ，φ(x)=elu(x)+1"""
    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.elu = nn.ELU()

    def forward(self, x, mask=None):
        # x: (B, K, D)
        Q = self.elu(self.to_q(x)) + 1
        K = self.elu(self.to_k(x)) + 1
        V = self.to_v(x)

        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=K.dtype)
            K = K * m
            V = V * m

        context = torch.einsum('bkd,bkv->bdv', K, V)               # (B, D, D)
        numerator = torch.einsum('bkd,bdv->bkv', Q, context)        # (B, K, D)
        denom = torch.einsum('bkd,bd->bk', Q, K.sum(dim=1)).unsqueeze(-1).clamp(min=1e-6)
        return numerator / denom


class LinearSelfAttention(nn.Module):
    """PreNorm 线性自注意力 + 残差。

    修正点:
      * PreNorm（LayerNorm 不再吃掉 residual 能量）
      * 正确的 linear-attention 归一化 (Σ φ(K))，去掉错误的 dim**-0.5
      * 移除 gain=0.1 的初始化（之前让 φ(x)+1 趋近常量，attention 退化为均值池化）
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.elu = nn.ELU()

    def forward(self, x, mask=None):
        """
        x: (B, K, D)
        mask: (B, K)  True for valid points
        """
        residual = x
        h = self.norm(x)
        Q = self.elu(self.to_q(h)) + 1
        K = self.elu(self.to_k(h)) + 1
        V = self.to_v(h)

        if mask is not None:
            valid = mask.unsqueeze(-1).to(dtype=K.dtype)
            K = K * valid
            V = V * valid

        context = torch.einsum('bkd,bkv->bdv', K, V)
        numerator = torch.einsum('bkd,bdv->bkv', Q, context)
        denom = torch.einsum('bkd,bd->bk', Q, K.sum(dim=1)).unsqueeze(-1).clamp(min=1e-6)
        out = numerator / denom
        return residual + out


class PointDualMatching1(nn.Module):
    def __init__(self, dim):
        """Point dual matching with stable linear self-attention."""
        super(PointDualMatching1, self).__init__()
        self.attn = LinearSelfAttention(dim)
        self.proj1 = nn.Linear(dim, dim, True)
        self.inf = np.inf

    def forward(self,
                ref_node_corr_knn_feats, src_node_corr_knn_feats,
                ref_node_corr_knn_scores, src_node_corr_knn_scores,
                ref_node_corr_knn_masks, src_node_corr_knn_masks):
        """
        ref_node_corr_knn_feats: (N, K, D)
        src_node_corr_knn_feats: (N, K, D)
        """
        # 局部自注意力增强
        ref_feats = self.attn(ref_node_corr_knn_feats, ref_node_corr_knn_masks)
        src_feats = self.attn(src_node_corr_knn_feats, src_node_corr_knn_masks)

        m_ref_feats = self.proj1(ref_feats)
        m_src_feats = self.proj1(src_feats)

        scores = torch.einsum('bnd,bmd->bnm', m_ref_feats, m_src_feats)  # (P, K, K)
        scores = scores / m_ref_feats.shape[-1] ** 0.5

        batch_size, num_row, num_col = scores.shape
        device = scores.device

        padded_row_masks = torch.zeros((batch_size, num_row, num_col), device=device)
        padded_row_masks.masked_fill_(~src_node_corr_knn_masks[:, None, :], float('-inf'))

        padded_col_masks = torch.zeros((batch_size, num_row, num_col), device=device)
        padded_col_masks.masked_fill_(~ref_node_corr_knn_masks[:, :, None], float('-inf'))

        matching_scores = (
            F.softmax(scores + padded_row_masks, dim=-1)
            * F.softmax(scores + padded_col_masks, dim=1)
        )
        matching_scores = (
            matching_scores
            * ref_node_corr_knn_scores[:, :, None]
            * src_node_corr_knn_scores[:, None, :]
        )

        return matching_scores


class PointDualMatching2(nn.Module):
    """Sinkhorn 最优传输双重匹配（替换简单 row/col softmax 乘积）。

    相比 PointDualMatching1：
    - 用可学习 Sinkhorn OT（SuperGlue 风格，带 dustbin）替换 softmax 乘积，
      强制每个点至多匹配一个对应点（双射约束），减少多对一误匹配。
    - dustbin 参数 alpha 可学习，自适应调节"无匹配"惩罚强度。
    """

    def __init__(self, dim: int, sinkhorn_iters: int = 3):
        super().__init__()
        self.attn = LinearSelfAttention(dim)
        self.proj1 = nn.Linear(dim, dim, bias=True)
        self.sinkhorn = LearnableLogOptimalTransport(num_iterations=sinkhorn_iters)

    def forward(
        self,
        ref_node_corr_knn_feats,   # (N, K, D)
        src_node_corr_knn_feats,   # (N, K, D)
        ref_node_corr_knn_scores,  # (N, K)
        src_node_corr_knn_scores,  # (N, K)
        ref_node_corr_knn_masks,   # (N, K)  bool, True=valid
        src_node_corr_knn_masks,   # (N, K)  bool, True=valid
    ):
        # 局部自注意力特征增强
        ref_feats = self.attn(ref_node_corr_knn_feats, ref_node_corr_knn_masks)
        src_feats = self.attn(src_node_corr_knn_feats, src_node_corr_knn_masks)

        m_ref = self.proj1(ref_feats)
        m_src = self.proj1(src_feats)

        # 缩放点积分数 (N, K, K)
        scores = torch.einsum('bnd,bmd->bnm', m_ref, m_src) / m_ref.shape[-1] ** 0.5

        # Sinkhorn OT → log 归一化概率 (N, K+1, K+1)，最后一行/列为 dustbin
        log_probs = self.sinkhorn(scores,
                                  row_masks=ref_node_corr_knn_masks,
                                  col_masks=src_node_corr_knn_masks)

        # 取非 dustbin 部分，转回概率
        matching_scores = log_probs[:, :-1, :-1].exp()

        # 乘以点置信度分数
        matching_scores = (
            matching_scores
            * ref_node_corr_knn_scores[:, :, None]
            * src_node_corr_knn_scores[:, None, :]
        )
        return matching_scores

    def __repr__(self):
        return self.__class__.__name__
