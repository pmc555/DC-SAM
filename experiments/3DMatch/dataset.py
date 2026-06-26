from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    """构建训练集和验证集数据加载器"""
    # ------------------------- 训练集配置 -------------------------
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,  # 数据集根目录路径（如'/data/3DMatch'）
        cfg.data.metadata_root,  # 元数据路径（包含预处理信息）
        'train',    # 使用训练集模式
        point_limit=cfg.train.point_limit, # 单点云最大点数限制（默认30000）
        use_augmentation=cfg.train.use_augmentation, # 是否启用数据增强（True/False
        augmentation_noise=cfg.train.augmentation_noise, # 高斯噪声标准差（0.005米）
        augmentation_rotation=cfg.train.augmentation_rotation, # 随机旋转幅度（1.0弧度≈57度）
        augmentation_crop=cfg.train.augmentation_crop, # 是否启用随机裁剪
        point_keep_ratio=cfg.train.point_keep_ratio, # 裁剪保留比例（0.7）
        matching_radius=cfg.train.matching_radius   # 真值匹配点对判定半径（0.1米）
    )
    # 构建训练数据加载器（多尺度模式）
    train_loader = build_dataloader_stack_mode(
        train_dataset,   # 训练数据集对象
        registration_collate_fn_stack_mode,  # 数据整理函数（处理多阶段特征堆叠
        cfg.backbone.num_stages,   # 主干网络阶段数（默认4）
        cfg.backbone.init_voxel_size, # 初始体素下采样尺寸（0.025米）
        cfg.backbone.num_neighbors,  # 各阶段K近邻数列表（如[35,35,35,35]）
        cfg.backbone.subsample_ratio,  # 下采样比率（默认2，每阶段分辨率减半）
        batch_size=cfg.train.batch_size,  # 批大小（默认1，点云对数量）
        num_workers=cfg.train.num_workers,  # 数据加载线程数（默认12）
        shuffle=True,   # 是否打乱数据顺序
        distributed=distributed,  # 是否分布式训练
        precompute_data=True  # 是否预计算邻接关系加速训练
    )
    # ------------------------- 验证集配置 -------------------------
    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        'val',  # 使用验证集模式
        point_limit=cfg.test.point_limit,  #测试时点数限制（None表示无限制）
        use_augmentation=cfg.train.use_augmentation,  # 继承训练集增强配置（通常为False）
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=False, # 验证集关闭裁剪增强
    )
    # 构建验证数据加载器（配置与训练集类似，但关闭shuffle）
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,  # 验证批大小（默认1）
        num_workers=cfg.test.num_workers,   # 验证线程数（默认8）
        shuffle=False,       # 验证集不打乱顺序
        distributed=distributed,
        precompute_data=True
    )

    return train_loader, valid_loader, cfg.backbone.num_neighbors # 返回加载器及邻域数列表


def test_data_loader(cfg, benchmark):
    """构建测试集数据加载器"""
    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,  # 测试集名称（如'3DLoMatch'）
        point_limit=cfg.test.point_limit, # 测试点数限制
        use_augmentation=False, # 测试集关闭所有数据增强
        augmentation_crop=False,  # 关闭裁剪
        rotated=True # 是否使用旋转版本数据集（用于鲁棒性测试）//False True
    )

    # 构建测试数据加载器（配置与验证集相同）
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,  # 测试批大小（默认1）
        num_workers=cfg.test.num_workers,  # 测试线程数（默认8）
        shuffle=False,   # 测试数据顺序固定
    )

    return test_loader, cfg.backbone.num_neighbors # 返回加载器及邻域数
