import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.ops import point_to_node_partition, index_select # 自定义点云分块和索引选择操作
from pareconv.modules.registration import get_node_correspondences  # 节点对应关系生成

from pareconv.modules.dual_matching import PointDualMatching1, PointDualMatching2  # 点级双重匹配模块

from pareconv.modules.geotransformer import (
    GeometricTransformer, # 几何变换器（自注意力+交叉注意力）
    SuperPointMatching,  # 超点匹配模块
    SuperPointTargetGenerator, # 超点目标生成器
)
from pareconv.modules.geotransformer.superpoint_matching import SuperPointMatching1
from pareconv.modules.geotransformer.geotransformer import ExtendedGeometricTransformer, ExtendedGeometricTransformer1
from pareconv.modules.registration import HypothesisProposer  # 假设生成器

from backbone import PAREConvFPN as _BackboneV0       # 原版，兼容预训练权重
from backboneV1 import PAREConvFPN as _BackboneV1    # 改进版（ChannelAggregationFFN），从头训练时用


class PARE_Net(nn.Module):
    def __init__(self, cfg):
        super(PARE_Net, self).__init__()  # 初始化配置参数
        self.num_points_in_patch = cfg.model.num_points_in_patch  # 每个块的点数
        self.matching_radius = cfg.model.ground_truth_matching_radius  # 匹配半径
        # 主干网络：use_v1=True 用改进版（训练），False 用原版（测试预训练权重）
        PAREConvFPN = _BackboneV1 if getattr(cfg.backbone, 'use_v1', False) else _BackboneV0
        self.backbone = PAREConvFPN(
            cfg.backbone.init_dim,  # 初始特征维度
            cfg.backbone.output_dim, # 输出特征维度
            cfg.backbone.kernel_size, # 卷积核尺寸
            cfg.backbone.share_nonlinearity, # 是否共享非线性层
            cfg.backbone.conv_way,# 卷积方式
            cfg.backbone.use_xyz, # 是否使用XYZ坐标
            cfg.fine_matching.use_encoder_re_feats# 是否使用编码器的旋转等变特征
        )
        # 几何变换器模块（用于特征增强）
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,# 输入维度
            cfg.geotransformer.output_dim,# 输出维度
            cfg.geotransformer.hidden_dim, # 隐藏层维度
            cfg.geotransformer.num_heads,  # 注意力头数
            cfg.geotransformer.blocks,  # Transformer块数
            cfg.geotransformer.sigma_d, # 距离缩放因子
            cfg.geotransformer.sigma_a, # 角度缩放因子
            cfg.geotransformer.angle_k, # 角度k近邻
            reduction_a=cfg.geotransformer.reduction_a,# 注意力缩减率
        )
        # 粗匹配目标生成器
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold # 目标数量，重叠阈值
        )
        #粗匹配模块
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization # 对应点数量，是否双重归一化
        )
        # self.coarse_matching1 = SuperPointMatching1(
        #     cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization, mnn_k=5
        # )
        # 细匹配假设生成器
        self.fine_matching = HypothesisProposer(
            cfg.fine_matching.topk,  # Top-K选择
            cfg.fine_matching.acceptance_radius,# 接受半径
            confidence_threshold=cfg.fine_matching.confidence_threshold, # 置信度阈值
            num_hypotheses=cfg.fine_matching.num_hypotheses,# 假设数量
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,# 优化步数
        )
        # 点级双重匹配模块：Sinkhorn OT（训练新模型）或 softmax（测试预训练权重）
        _dim = cfg.backbone.output_dim // 3 * 3
        if getattr(cfg.fine_matching, 'use_sinkhorn', False):
            self.point_matching = PointDualMatching2(dim=_dim)
        else:
            self.point_matching = PointDualMatching1(dim=_dim)

    def forward(self, data_dict):
        output_dict = {}
        # Downsample point clouds
        # ------------ 数据预处理 ------------
        # 获取输入特征和变换矩阵
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        # 获取不同层级的点云长度信息
        ref_length_c = data_dict['lengths'][-1][0].item() # 最粗层参考点云长度
        ref_length_f = data_dict['lengths'][1][0].item() # 中间层参考点云长度
        ref_length = data_dict['lengths'][0][0].item() # 原始层参考点云长度
        points_c = data_dict['points'][-1].detach()# 最粗层点云
        points_f = data_dict['points'][1].detach() # 中间层点云
        points = data_dict['points'][0].detach() # 原始层点云

        # 分割参考点云和源点云
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        # 将各层点云存入输出字典
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        # 1. Generate ground truth node correspondences 生成真实节点对应关系
        # 使用point_to_node_partition进行点云分块
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )  # ref_N_c,  [ref_N_c, 64],  [ref_N_c, 64],
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        output_dict['ref_node_knn_indices'] = ref_node_knn_indices
        output_dict['src_node_knn_indices'] = src_node_knn_indices
        # 填充点云防止越界
        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)# [ref_N_f + 1, 3]
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        # 获取节点KNN点坐标
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0) #[ref_N_c, 64, 3]
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        # 生成真实节点对应关系
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )  # coarse correspondences  gt_node_corr_indices: [N, 2]  gt_node_corr_overlaps : N


        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. PARE-Conv Encoder 主干网络特征提取
        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(data_dict)# 通过PARE-Conv FPN提取多层级特征

        # 3. Conditional Transformer 几何变换器处理
        # 分割参考和源点云特征
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        ref_feats_c_re = re_feats_c[:ref_length_c]
        src_feats_c_re = re_feats_c[ref_length_c:]
        output_dict['ref_feats_c_re'] = ref_feats_c_re
        output_dict['src_feats_c_re'] = src_feats_c_re
        # ...处理旋转等变特征..
        # 通过几何变换器增强特征

        # print(f"ref_feats_c:{ref_feats_c.shape}")
        # print(f"src_feats_c:{src_feats_c.shape}")
        ref_feats_c, src_feats_c, scores_list = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )

        # 特征归一化
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 4. Head for fine level matching 细匹配层级处理
          # 分割细粒度特征
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        # print(f"ref_feats_f:{ref_feats_f.shape}")
        # print(f"src_feats_f:{src_feats_f.shape}")
        m_ref_scores = m_scores[:ref_length_f]
        m_src_scores = m_scores[ref_length_f:]
        re_ref_feats_f = re_feats_f[:ref_length_f]
        re_src_feats_f = re_feats_f[ref_length_f:]

        output_dict['m_ref_scores'] = m_ref_scores
        output_dict['m_src_scores'] = m_src_scores
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        output_dict['re_ref_feats_f'] = re_ref_feats_f
        output_dict['re_src_feats_f'] = re_src_feats_f
        # ...处理匹配分数...

        # 5. Select topk nearest node correspondences 粗匹配，选择Top-K节点对应关系
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            # 7 Random select ground truth node correspondences during training  # 训练时使用真实对应关系
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 6 Generate batched node points & feats 生成批处理节点数据
        #获取对应节点的KNN索引和点坐标
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)
        # 填充特征防止越界
        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        # 提取对应节点的KNN特征
        ref_node_corr_knn_scores = index_select(m_ref_padded_scores, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_scores = index_select(m_src_padded_scores, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points   # 256 64 3
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        re_ref_padded_feats_f = torch.cat([re_ref_feats_f, torch.zeros_like(re_ref_feats_f[:1])], dim=0)
        re_src_padded_feats_f = torch.cat([re_src_feats_f, torch.zeros_like(re_src_feats_f[:1])], dim=0)
        re_ref_node_corr_knn_feats = index_select(re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        re_src_node_corr_knn_feats = index_select(re_src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['re_ref_node_corr_knn_feats'] = re_ref_node_corr_knn_feats   # 256 64 21 3
        output_dict['re_src_node_corr_knn_feats'] = re_src_node_corr_knn_feats

        # 7 Match batched points 点级匹配
        matching_scores = self.point_matching(ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores   # 256 64 64
        output_dict['ref_node_corr_knn_scores'] = ref_node_corr_knn_scores
        output_dict['src_node_corr_knn_scores'] = src_node_corr_knn_scores



        # 8 Generate hypotheses and select the best one 假设生成与优化
        with torch.no_grad():
            # 生成最终变换假设
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, re_ref_corr_feats, re_src_corr_feats, = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                re_ref_node_corr_knn_feats,
                re_src_node_corr_knn_feats,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
            )
        # 存储最终结果
        output_dict['re_ref_corr_feats'] = re_ref_corr_feats
        output_dict['re_src_corr_feats'] = re_src_corr_feats
        output_dict['hypotheses'] = hypotheses
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
        output_dict['transform'] = transform

        return output_dict

# ------------ 辅助函数 ------------
def create_model(config):
    model = PARE_Net(config)
    return model

def main():
    """测试用主函数"""
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
