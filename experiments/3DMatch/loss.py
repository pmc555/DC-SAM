import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss # 加权环形损失
from pareconv.modules.ops.transformation import apply_transform # 点云变换应用函数
from pareconv.modules.registration.metrics import isotropic_transform_error, relative_rotation_error  # 变换误差计算
from pareconv.modules.ops.pairwise_distance import pairwise_distance # 成对距离计算


class CoarseMatchingLoss(nn.Module):
    """粗匹配阶段损失函数"""
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        # 初始化加权环形损失（带边界参数）
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin, # 正样本边界 0.1
            cfg.coarse_loss.negative_margin,  # 负样本边界 1.4
            cfg.coarse_loss.positive_optimal, # 正样本最优距离 0.1
            cfg.coarse_loss.negative_optimal, # 负样本最优距离 1.4
            cfg.coarse_loss.log_scale,  # 对数缩放因子 24
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap # 正样本重叠阈值 0.1

    def forward(self, output_dict):
        # 获取输入数据
        ref_feats = output_dict['ref_feats_c'] # 参考点云超点特征 [B,N,D]
        src_feats = output_dict['src_feats_c'] # 源点云超点特征 [B,M,D]
        gt_node_corr_indices = output_dict['gt_node_corr_indices']  # 真值对应索引 [K,2]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps'] # 重叠率 [K]
        # 提取真值对应的行列索引
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0] # 参考点云有效索引 [K]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1] # 源点云有效索引 [K]
        # 计算特征间L2距离矩阵 [N,M]；clamp 防止 sqrt(0) 反传出 inf 梯度
        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True).clamp(min=1e-12))
        # 构建重叠率矩阵 [B,N,M]
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        # 生成正负样本掩码
        pos_masks = torch.gt(overlaps, self.positive_overlap) # 正样本：重叠率>0.1
        neg_masks = torch.eq(overlaps, 0)   # 负样本：无重叠
        pos_scales = torch.sqrt(overlaps * pos_masks.float())   # 加权系数（重叠率平方根）
        # 计算加权环形损失
        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss # 标量损失值

class FineMatchingLoss(nn.Module): # for fine dual matching
    """细粒度匹配损失（含旋转不变/等变双路损失）"""
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        # 课程学习的终点值（config 中配置）
        self.positive_radius_init = cfg.fine_loss.positive_radius  # 0.05m
        self.negative_radius_init = cfg.fine_loss.negative_radius  # 0.20m
        # 训练起始时用 2× 宽松半径，随 epoch 收紧到 init 值
        self.positive_radius = self.positive_radius_init * 2.0
        self.negative_radius = self.negative_radius_init * 2.0
        self.positive_margin = cfg.fine_loss.positive_margin  # 0.1
        self.negative_margin = cfg.fine_loss.negative_margin  # 1.4

    def forward(self, output_dict, data_dict):
        # 获取点云数据与变换矩阵
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']# 参考点云局部块点 [B,K1,3]
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']  # 源点云局部块点 [B,K2,3]
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']  # 参考点有效掩码 [B,K1]
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks'] # 源点有效掩码 [B,K2]
        ref_node_corr_knn_scores = output_dict['ref_node_corr_knn_scores']
        src_node_corr_knn_scores = output_dict['src_node_corr_knn_scores']

        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform) # 应用真值变换将源点云对齐到参考坐标系
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)  # 计算点对距离矩阵 [B,K1,K2]
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1)) # 生成有效点对掩码（双方点都有效）
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)  # 生成真值对应关系（距离<0.05m视为正样本）
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        # 处理单边无效点（无对应点的行/列）
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks) # 行无对应
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)  # 列无对应

        # compute matching loss of rotation invariant features ------ 旋转不变特征匹配损失 ------
        # nan_to_num before clamp: NaN passes through .clamp(min=eps) unchanged,
        # so sinkhorn fp16 overflow → matching_scores NaN → log → NaN propagates
        # and freezes the entire model under AMP GradScaler.  Replace NaN→eps,
        # +inf→1, -inf→eps before clamp so log() stays finite.
        eps = 1e-8
        device = matching_scores.device
        ms_safe = torch.nan_to_num(matching_scores, nan=eps, posinf=1.0, neginf=eps)
        ref_safe = torch.nan_to_num(ref_node_corr_knn_scores, nan=1.0 - eps, posinf=1.0 - eps, neginf=eps)
        src_safe = torch.nan_to_num(src_node_corr_knn_scores, nan=1.0 - eps, posinf=1.0 - eps, neginf=eps)
        pos_logp = ms_safe[gt_corr_map].clamp(min=eps).log()
        ri_pos = pos_logp.mean() if pos_logp.numel() > 0 else torch.tensor(0.0, device=device)
        row_logp = (1 - ref_safe)[slack_row_labels].clamp(min=eps).log()
        ri_row = row_logp.mean() if row_logp.numel() > 0 else torch.tensor(0.0, device=device)
        col_logp = (1 - src_safe)[slack_col_labels].clamp(min=eps).log()
        ri_col = col_logp.mean() if col_logp.numel() > 0 else torch.tensor(0.0, device=device)
        fine_ri_loss = -(ri_pos + 0.5 * ri_row + 0.5 * ri_col)

        # compute loss of rotation equivariant features------ 旋转等变特征对齐损失 ------
        # 生成负样本掩码（距离>0.2m）
        neg_map = torch.gt(dists, self.negative_radius ** 2)
        neg_map = torch.logical_and(neg_map, gt_masks)
        fine_re_loss = self.fine_re_loss(output_dict, gt_corr_map, neg_map, transform)

        return fine_ri_loss, fine_re_loss

    def anneal(self, epoch: int, max_epoch: int):
        """课程学习：随训练进度逐渐收紧匹配半径判定。

        epoch=0  → positive_radius = 2× init（宽松，降低初期训练难度）
        epoch=T  → positive_radius = init（目标精度）
        negative_radius 同步线性退火。
        """
        progress = min(1.0, epoch / max(1, max_epoch))
        self.positive_radius = self.positive_radius_init * (2.0 - progress)
        self.negative_radius = self.negative_radius_init * (2.0 - progress)

    def fine_re_loss(self, out_dict, gt_corr_map, neg_map, gt_trans):
        """旋转等变特征对比损失"""
        # 获取旋转等变特征
        ref_feats = out_dict['re_ref_node_corr_knn_feats'] # [B,K1,C,3]
        src_feats = out_dict['re_src_node_corr_knn_feats'] # [B,K2,C,3]
        # Guard: VN equivariant features can overflow to NaN/inf for degenerate
        # point-cloud pairs under fp16 AMP.  Return 0 instead of propagating NaN
        # to total loss (allows c_loss + f_ri_loss to update every batch).
        if not (torch.isfinite(ref_feats).all() and torch.isfinite(src_feats).all()):
            return torch.tensor(0.0, device=ref_feats.device)
        # 正样本处理
        batch_indices, ref_indices, src_indices = torch.nonzero(gt_corr_map, as_tuple=True)
        if batch_indices.shape[0] == 0:
            return torch.tensor(0.0, device=ref_feats.device)
        ref_feats_rot = ref_feats[batch_indices, ref_indices]
        src_feats_rot = src_feats[batch_indices, src_indices]
        src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_trans[:3, :3]) # 将源特征旋转到参考坐标系
        pos_loss = torch.relu(torch.norm(src_feats_rot - ref_feats_rot, 2, -1) - self.positive_margin).mean() # 计算正样本特征距离
        # 负样本处理
        batch_indices, ref_indices, src_indices = torch.nonzero(neg_map, as_tuple=True)
        if batch_indices.shape[0] == 0:
            neg_loss = torch.tensor(0.0, device=ref_feats.device)
        else:
            ref_feats_rot = ref_feats[batch_indices, ref_indices]
            src_feats_rot = src_feats[batch_indices, src_indices]
            src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_trans[:3, :3])
            neg_loss = torch.relu(self.negative_margin - torch.norm(src_feats_rot - ref_feats_rot, 2, -1)).mean()
        re_loss = pos_loss + neg_loss
        return re_loss

class OverallLoss(nn.Module):
    """多任务损失整合器（Kendall et al. 2018 不确定性自适应加权）。

    每个子任务的权重由可学习 log-variance 参数控制：
        L_total = Σ_i [ exp(-log_s_i) * L_i + 0.5 * log_s_i ]
    log_s_i 初始化为 0（对应权重=1），在训练中自适应调节各任务权重。
    """
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        # 可学习 log-variance（初始 0 → 权重=exp(0)=1.0；正数 → 降权；负数 → 升权）
        self.log_s_coarse = nn.Parameter(torch.zeros(1))
        self.log_s_ri = nn.Parameter(torch.zeros(1))
        self.log_s_re = nn.Parameter(torch.zeros(1))

    def anneal(self, epoch: int, max_epoch: int):
        """代理 FineMatchingLoss 的课程退火。"""
        self.fine_loss.anneal(epoch, max_epoch)

    def forward(self, output_dict, data_dict):
        # 计算各阶段损失
        coarse_loss = self.coarse_loss(output_dict)
        fine_ri_loss, fine_re_loss = self.fine_loss(output_dict, data_dict)
        # Kendall et al. 不确定性加权：exp(-log_s) * L + 0.5 * log_s
        loss = (
            torch.exp(-self.log_s_coarse) * coarse_loss + 0.5 * self.log_s_coarse +
            torch.exp(-self.log_s_ri)     * fine_ri_loss + 0.5 * self.log_s_ri +
            torch.exp(-self.log_s_re)     * fine_re_loss + 0.5 * self.log_s_re
        )
        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_ri_loss': fine_ri_loss,
            'f_re_loss': fine_re_loss,
            # 监控权重（log_s→0 表示高置信；log_s>0 表示该任务降权）
            'w_coarse': torch.exp(-self.log_s_coarse).detach(),
            'w_ri': torch.exp(-self.log_s_ri).detach(),
            'w_re': torch.exp(-self.log_s_re).detach(),
        }


class Evaluator(nn.Module):
    """多指标评估器"""
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap # 重叠接受阈值 0.0
        self.acceptance_radius = cfg.eval.acceptance_radius  # 内点半径 0.1m
        self.acceptance_rmse = cfg.eval.rmse_threshold  # RMSE阈值 0.2m
        self.feat_rre_threshold = cfg.eval.feat_rre_threshold  # 特征旋转误差阈值 20度

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        """粗匹配精度评估（PIR: Patch Inlier Ratio）。

        用 1D 哈希 (ref_idx * src_length_c + src_idx) + torch.isin 取代 N*M 稠密索引矩阵，
        典型规模下显存从 O(N*M) 降到 O(K + P)。
        """
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        if ref_node_corr_indices.numel() == 0:
            return torch.tensor(0.0, device=gt_node_corr_overlaps.device)

        gt_keys = gt_node_corr_indices[:, 0].long() * src_length_c + gt_node_corr_indices[:, 1].long()
        pred_keys = ref_node_corr_indices.long() * src_length_c + src_node_corr_indices.long()
        precision = torch.isin(pred_keys, gt_keys).float().mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        """细粒度内点率（IR: Inlier Ratio）"""
        # 应用估计变换
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']

        if src_corr_points.shape[0] == 0:
            return torch.tensor(0.0, device=transform.device)
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        mask = torch.lt(corr_distances, self.acceptance_radius)
        precision = mask.float().mean()
        return precision
    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        """配准指标计算（RRE, RTE, RMSE, RR）"""
        # 计算变换误差
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        # 计算配准后误差
        src_points = output_dict['src_points']
        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        # 判断是否成功（RMSE<0.2m）
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)
        """全指标评估"""
        return {
            'PIR': c_precision,  # 粗匹配精度
            'IR': f_precision, # 细匹配内点率
            'RRE': rre, # 旋转误差（度）
            'RTE': rte, # 平移误差（米）
            'RMSE': rmse,  # 均方根误差
            'RR': recall, # 配准召回率
        }
