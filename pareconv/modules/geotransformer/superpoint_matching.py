import torch
import torch.nn as nn

from pareconv.modules.ops import pairwise_distance

class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores

import torch
import torch.nn as nn
from pareconv.modules.ops import pairwise_distance


class SuperPointMatching1(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True, mnn_k=10):
        super(SuperPointMatching1, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
        self.mnn_k = mnn_k  # 0=不用MNN, 1=严格MNN, >1=互为k-NN


    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        device = ref_feats.device
        if ref_masks is None:
            ref_masks = torch.ones(ref_feats.shape[0], dtype=torch.bool, device=device)
        if src_masks is None:
            src_masks = torch.ones(src_feats.shape[0], dtype=torch.bool, device=device)

        # ---- 有效索引（全局->局部） ----
        ref_idx_global = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_idx_global = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats_l = ref_feats[ref_idx_global]  # [Nr, C]
        src_feats_l = src_feats[src_idx_global]  # [Ns, C]



        dists = pairwise_distance(ref_feats_l, src_feats_l, normalized=True)  # [Nr, Ns]
        scores = torch.exp(-dists)

        if self.dual_normalization:
            scores = (scores / (scores.sum(dim=1, keepdim=True) + 1e-8)) * \
                     (scores / (scores.sum(dim=0, keepdim=True) + 1e-8))

        Nr, Ns = scores.shape
        num_corr = min(self.num_correspondences, Nr * Ns)

        # ---- 全局Top-K（局部索引）----
        flat_scores = scores.reshape(-1)
        topk_scores, topk_flat_idx = torch.topk(flat_scores, k=num_corr, largest=True)
        ref_sel_local = topk_flat_idx // Ns  # [K] in [0, Nr)
        src_sel_local = topk_flat_idx %  Ns  # [K] in [0, Ns)

        # ---- k-NN Mutual（向量化，完全在“局部索引空间”）----
        if self.mnn_k > 0 and Nr > 0 and Ns > 0:
            k_ref = min(self.mnn_k, Ns)
            k_src = min(self.mnn_k, Nr)

            # ref->src 的 top-k：shape [Nr, k_ref]
            _, ref_topk_idx = torch.topk(scores, k=k_ref, dim=1)
            # src->ref 的 top-k：shape [k_src, Ns]（注意dim=0）
            _, src_topk_idx = torch.topk(scores, k=k_src, dim=0)

            # 构造布尔掩码：mask_ref[i,j]=True 表示 j 在 ref i 的 top-k 里
            mask_ref = torch.zeros_like(scores, dtype=torch.bool)
            mask_ref.scatter_(1, ref_topk_idx, True)  # dim=1 按列散射

            # mask_src[i,j]=True 表示 i 在 src j 的 top-k 里（注意 dim=0）
            mask_src = torch.zeros_like(scores, dtype=torch.bool)
            mask_src.scatter_(0, src_topk_idx, True)  # dim=0 按行散射

            mutual_mask_2d = mask_ref & mask_src  # [Nr, Ns]

            # 只保留 Top-K 里的那些互为k-NN的对
            keep = mutual_mask_2d[ref_sel_local, src_sel_local]  # [K] 布尔
            # 若过严导致全被筛掉，可选择不筛或降级
            if keep.any():
                ref_sel_local = ref_sel_local[keep]
                src_sel_local = src_sel_local[keep]
                topk_scores   = topk_scores[keep]
            # 否则保留原 Top-K（回退），避免空集合导致后续指标全0/NaN

        # ---- 映射回原始全局索引 ----
        ref_corr_indices = ref_idx_global[ref_sel_local]
        src_corr_indices = src_idx_global[src_sel_local]

        return ref_corr_indices, src_corr_indices, topk_scores

class SuperPointMatchingOptimized(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True, mnn_k=1):
        super(SuperPointMatchingOptimized, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
        self.mnn_k = mnn_k  # 初始为严格 MNN

    @torch.no_grad()
    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None, alpha=0.5):
        device = ref_feats.device
        if ref_masks is None:
            ref_masks = torch.ones(ref_feats.shape[0], dtype=torch.bool, device=device)
        if src_masks is None:
            src_masks = torch.ones(src_feats.shape[0], dtype=torch.bool, device=device)

        # 有效索引
        ref_idx_global = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_idx_global = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats_l = ref_feats[ref_idx_global]
        src_feats_l = src_feats[src_idx_global]

        # ---- 相似度（基于 cosine 相似度，无温度项）----
        cos_sim = 1 - pairwise_distance(ref_feats_l, src_feats_l, normalized=True)
        scores = torch.clamp((cos_sim + 1.0) / 2.0, 0.0, 1.0)  # [0,1]

        # ---- 可调的双归一化 ----
        if self.dual_normalization:
            ref_norm = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
            src_norm = scores / (scores.sum(dim=0, keepdim=True) + 1e-8)
            scores = (1 - alpha) * ref_norm + alpha * (ref_norm * src_norm)

        Nr, Ns = scores.shape
        num_corr = min(self.num_correspondences, Nr * Ns)

        # ---- Top-K ----
        flat_scores = scores.reshape(-1)
        topk_scores, topk_flat_idx = torch.topk(flat_scores, k=num_corr, largest=True)
        ref_sel_local = topk_flat_idx // Ns
        src_sel_local = topk_flat_idx % Ns

        # ---- MNN 互查 ----
        if self.mnn_k > 0 and Nr > 0 and Ns > 0:
            k_ref = min(self.mnn_k, Ns)
            k_src = min(self.mnn_k, Nr)

            _, ref_topk_idx = torch.topk(scores, k=k_ref, dim=1)
            _, src_topk_idx = torch.topk(scores, k=k_src, dim=0)

            mask_ref = torch.zeros_like(scores, dtype=torch.bool)
            mask_ref.scatter_(1, ref_topk_idx, True)
            mask_src = torch.zeros_like(scores, dtype=torch.bool)
            mask_src.scatter_(0, src_topk_idx, True)

            mutual_mask_2d = mask_ref & mask_src
            keep = mutual_mask_2d[ref_sel_local, src_sel_local]

            if keep.any():
                ref_sel_local = ref_sel_local[keep]
                src_sel_local = src_sel_local[keep]
                topk_scores   = topk_scores[keep]

        # ---- 映射回原始索引 ----
        ref_corr_indices = ref_idx_global[ref_sel_local]
        src_corr_indices = src_idx_global[src_sel_local]

        return ref_corr_indices, src_corr_indices, topk_scores

    def update_mnn_k(self, epoch, max_epoch):
        """动态调整 MNN 严格程度"""
        ratio = epoch / max_epoch
        if ratio < 0.3:
            self.mnn_k = 1
        elif ratio < 0.7:
            self.mnn_k = 3
        else:
            self.mnn_k = 0

