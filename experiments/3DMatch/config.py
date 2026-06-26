import os
import os.path as osp
import argparse

from easydict import EasyDict as edict  # 用于创建嵌套字典结构

from pareconv.utils.common import ensure_dir  # 确保目录存在的工具函数

# 创建基础配置字典
_C = edict()

# common------------------------- 基础配置 -------------------------
_C.seed = 7351  # 全局随机种子

# dirs ------------------------- 目录配置 -------------------------
_C.working_dir = osp.dirname(osp.realpath(__file__)) # 当前文件所在目录
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))  # 项目根目录
_C.exp_name = osp.basename(_C.working_dir)  # 实验名称（当前目录名）
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name) # 总输出目录
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots') # 模型快照保存路径
_C.log_dir = osp.join(_C.output_dir, 'logs')  # 训练日志路径
_C.event_dir = osp.join(_C.output_dir, 'wandb_events') # wandb事件记录路径
_C.feature_dir = osp.join(_C.output_dir, 'features')  # 特征保存路径
_C.registration_dir = osp.join(_C.output_dir, 'registration')  # 配准结果保存路径
# 确保所有目录存在
ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data------------------------- 数据集配置 -------------------------
_C.data = edict()
_C.data.dataset_root = 'F:/dataset/PUBLIC_3D/3DMatch'  # 数据集根目录（含 train/ test/ train_pair_overlap_masks/）
_C.data.metadata_root = osp.join(_C.root_dir, 'data', '3DMatch', 'metadata') # 元数据路径

# train data------------------------- 训练配置 -------------------------
_C.train = edict()
_C.train.batch_size = 1 # 批处理大小（点云对数量）
_C.train.num_workers = 12    # 数据加载线程数（原始为12，可能根据硬件调整）
_C.train.point_limit = 30000  # 单点云最大点数限制（初始30000）
_C.train.use_augmentation = True  # 是否使用数据增强
_C.train.augmentation_noise = 0.005  # 添加高斯噪声的标准差（米）
_C.train.augmentation_rotation = 1.0 # 随机旋转范围（弧度）
_C.train.augmentation_crop = True  # 是否启用随机裁剪
_C.train.point_keep_ratio = 0.7  # 裁剪保留比例
_C.train.matching_radius = 0.1  # 匹配点对的真值判定半径（米）

# test data------------------------- 测试配置 -------------------------
_C.test = edict()
_C.test.batch_size = 1  # 测试批大小
_C.test.num_workers = 8 # 测试数据加载线程数
_C.test.point_limit = None # 测试时点数限制（None表示不限制）

# evaluation ------------------------- 评估指标 -------------------------
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0 # 接受配准的最小重叠率
_C.eval.acceptance_radius = 0.1  # 内点判定半径
_C.eval.inlier_ratio_threshold = 0.05  # 有效匹配比率阈值
_C.eval.rmse_threshold = 0.2  # RMSE成功阈值（米）
_C.eval.rre_threshold = 15.0 # 相对旋转误差阈值（度）
_C.eval.rte_threshold = 0.3 # 相对平移误差阈值（米）
_C.eval.feat_rre_threshold = 20.0  # 特征级旋转误差阈值

# ransac------------------------- RANSAC配置 -------------------------
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05 # 内点距离阈值
_C.ransac.num_points = 3 # 每次采样点对数量
_C.ransac.num_iterations = 1000 # 最大迭代次数

# optim------------------------- 优化器配置 -------------------------
_C.optim = edict()
_C.optim.lr = 1e-4 # 初始学习率
_C.optim.lr_decay = 0.95 # 学习率衰减系数
_C.optim.lr_decay_steps = 1 # 衰减步长（每epoch衰减）
_C.optim.weight_decay = 1e-6  # 权重衰减系数
_C.optim.max_epoch = 40  # 最大训练轮次
_C.optim.grad_acc_steps = 1 # 梯度累积步数

# model - backbone ------------------------- 主干网络配置 -------------------------
_C.backbone = edict()
_C.backbone.num_stages = 4 # 下采样阶段数
_C.backbone.num_neighbors = [35] * _C.backbone.num_stages  # we use constant neighbors # 各阶段KNN邻域点数
_C.backbone.init_voxel_size = 0.025  # 初始体素下采样尺寸（米）
_C.backbone.subsample_ratio = 2  # 下采样比率
_C.backbone.kernel_size = 4  # PARE-Conv核数量
_C.backbone.share_nonlinearity = False # 是否共享非线性层
_C.backbone.conv_way = 'edge_conv'  # 'edge_conv' or 'node_conv'# 卷积方式：edge_conv/node_conv
_C.backbone.use_xyz = True # 是否在特征中包含坐标信息
_C.backbone.init_dim = 96 # 初始特征维度
_C.backbone.output_dim = 256 # 输出特征维度
_C.backbone.use_v1 = True   # True=backboneV1(ChannelAggregationFFN, 从头训练); False=原版(兼容pretrained)

# model - Global------------------------ 全局模型配置 -------------------------
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05 # 真值匹配半径
_C.model.num_points_in_patch = 128 # 每个局部块的点数

# model - Coarse Matching ------------------------- 粗匹配配置 -------------------------
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128 # 目标匹配对数
_C.coarse_matching.overlap_threshold = 0.1  # 重叠区域阈值
_C.coarse_matching.num_correspondences = 256 # 粗匹配候选数
_C.coarse_matching.dual_normalization = True  # 是否双重归一化

# model - GeoTransformer ------------------------- 几何变换器配置 -------------------------
_C.geotransformer = edict()
_C.geotransformer.input_dim = 768 # 输入特征维度
_C.geotransformer.hidden_dim = 192  # 隐层维度
_C.geotransformer.output_dim = 192 # 输出维度
_C.geotransformer.num_heads = 4  # 注意力头数
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']  # 模块结构
_C.geotransformer.sigma_d = 0.1  # 距离特征缩放因子
_C.geotransformer.sigma_a = 15 # 角度特征缩放因子
_C.geotransformer.angle_k = 3 # 角度计算K近邻数
_C.geotransformer.reduction_a = 'max'  # 注意力聚合方式

# model - Fine Matching ------------------------- 细匹配配置 -------------------------
_C.fine_matching = edict()
_C.fine_matching.topk = 3  # Top-K候选选择数
_C.fine_matching.acceptance_radius = 0.1  # 内点接受半径
_C.fine_matching.confidence_threshold = 0.005 # 置信度阈值
_C.fine_matching.num_hypotheses = 5000  # 生成假设数量
_C.fine_matching.num_refinement_steps = 5 # 优化迭代次数
_C.fine_matching.use_encoder_re_feats = True # 是否使用编码器旋转等变特征
_C.fine_matching.use_sinkhorn = True  # True=PointDualMatching2(Sinkhorn OT); False=PointDualMatching1(softmax)

# loss - Coarse level------------------------- 损失函数配置 -------------------------
_C.coarse_loss = edict() # 粗匹配损失
_C.coarse_loss.positive_margin = 0.1 # 正样本间隔
_C.coarse_loss.negative_margin = 1.4  # 负样本间隔
_C.coarse_loss.positive_optimal = 0.1 # 正样本最优距离
_C.coarse_loss.negative_optimal = 1.4  # 负样本最优距离
_C.coarse_loss.log_scale = 24 # 对数缩放系数
_C.coarse_loss.positive_overlap = 0.1   # 正样本重叠阈值

# loss - Fine level
_C.fine_loss = edict() # 细匹配损失
_C.fine_loss.positive_radius = 0.05  # 正样本半径（课程学习的终点值）
_C.fine_loss.negative_radius = 0.2  # 负样本半径（课程学习的终点值）
_C.fine_loss.positive_margin = 0.1  # 正样本间隔
_C.fine_loss.negative_margin = 1.4   # 负样本间隔
_C.fine_loss.curriculum = True  # True=启用匹配半径课程退火 (0.10→0.05m)

# loss - Overall
_C.loss = edict() # 总损失权重
_C.loss.weight_coarse_loss = 1.0 # 粗匹配损失权重
_C.loss.weight_fine_ri_loss = 1.0  # 细匹配旋转不变损失权重
_C.loss.weight_fine_re_loss = 1.0 # 细匹配旋转等变损失权重

# ------------------------- 辅助函数 -------------------------
def make_cfg():
    """返回配置字典的深拷贝"""
    return _C


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    """主函数：处理符号链接创建"""
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')  # 创建软链接方便访问


if __name__ == '__main__':
    main()
