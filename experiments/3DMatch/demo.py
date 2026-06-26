import argparse

import torch
import numpy as np
import time
from pareconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from pareconv.utils.torch import to_cuda, release_cuda
from pareconv.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from pareconv.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
from thop import profile  # 新增：用于计算FLOP

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default='/userData/gpuzsh/parenet/datasets/3DMatch/test/7-scenes-redkitchen/cloud_bin_5.pth', help="src point cloud numpy file")
    parser.add_argument("--ref_file", default='/userData/gpuzsh/parenet/datasets/3DMatch/test/7-scenes-redkitchen/cloud_bin_0.pth', help="src point cloud numpy file")
    parser.add_argument("--gt_file", default='../../data/demo/gt.npy', help="ground-truth transformation file")
    parser.add_argument("--weights", default='/userData/gpuzsh/parenet/output/3DMatch/snapshots/20251108-055511/epoch-39.pth.tar', help="model weights file")
    # parser.add_argument("--weights", default='../../pretrain/3dmatch.pth.tar', help="model weights file")
    parser.add_argument("--vis_patches", action="store_true", help="visualize point cloud patches")
    parser.add_argument(
        "--vis_patch_level",
        default="fine",
        choices=["fine", "coarse"],
        help="patch level used for visualization",
    )
    parser.add_argument("--num_vis_patches", type=int, default=128, help="max patch count to visualize")
    parser.add_argument("--vis_seed", type=int, default=0, help="random seed for patch colors")
    return parser


def load_points(file):
    if file.endswith(".npy"):
        data = np.load(file)
    elif file.endswith(".pth"):
        data = torch.load(file, weights_only=False)
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, dict):
            # 如果 .pth 里存的是 dict，点云通常在 'points'
            data = data.get("points", None)
            if data is None:
                raise ValueError(f"Cannot find 'points' in {file}")
            data = data.cpu().numpy()
    else:
        raise ValueError(f"Unsupported file type: {file}")

    return data.astype(np.float32)
# --------------------------------------------------------
def load_data(args):
    src_points = load_points(args.src_file)
    ref_points = load_points(args.ref_file)

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def _tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _make_patch_colors(points, node_knn_indices, max_patches, seed):
    points = _tensor_to_numpy(points)
    node_knn_indices = _tensor_to_numpy(node_knn_indices)
    num_points = points.shape[0]
    num_nodes = node_knn_indices.shape[0]
    num_vis_nodes = min(max_patches, num_nodes)

    point_patch_ids = np.full(num_points, fill_value=-1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    patch_colors = rng.uniform(0.15, 1.0, size=(num_vis_nodes, 3)).astype(np.float32)

    for patch_id in range(num_vis_nodes):
        patch_indices = node_knn_indices[patch_id]
        patch_indices = patch_indices[(patch_indices >= 0) & (patch_indices < num_points)]
        point_patch_ids[patch_indices] = patch_id

    colors = np.full((num_points, 3), fill_value=0.65, dtype=np.float32)
    valid_mask = point_patch_ids >= 0
    colors[valid_mask] = patch_colors[point_patch_ids[valid_mask]]
    return colors, num_vis_nodes


def visualize_patches(output_dict, args):
    if args.vis_patch_level == "fine":
        ref_points = output_dict["ref_points_f"]
        src_points = output_dict["src_points_f"]
    else:
        ref_points = output_dict["ref_points_c"]
        src_points = output_dict["src_points_c"]

    ref_indices = output_dict["ref_node_knn_indices"]
    src_indices = output_dict["src_node_knn_indices"]
    ref_colors, ref_vis_count = _make_patch_colors(ref_points, ref_indices, args.num_vis_patches, args.vis_seed)
    src_colors, src_vis_count = _make_patch_colors(src_points, src_indices, args.num_vis_patches, args.vis_seed + 1)

    ref_points = _tensor_to_numpy(ref_points)
    src_points = _tensor_to_numpy(src_points)
    offset = np.array([ref_points[:, 0].max() - src_points[:, 0].min() + 0.6, 0.0, 0.0], dtype=np.float32)
    src_points_vis = src_points + offset

    ref_pcd = make_open3d_point_cloud(ref_points, colors=ref_colors)
    src_pcd = make_open3d_point_cloud(src_points_vis, colors=src_colors)
    draw_geometries(ref_pcd, src_pcd)
    print(
        f"Patch visualization done ({args.vis_patch_level}): "
        f"ref={ref_vis_count} patches, src={src_vis_count} patches."
    )


# 新增：计算模型参数量的函数
def count_model_parameters(model):
    """计算模型参数量，返回(M)和总参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6  # 转换为M（百万）
    return params_m, total_params


# 新增：计算推理时间的函数
def calculate_inference_latency(model, input_data, warmup=5, repeat=20):
    """计算模型推理时间（ms），包含预热和多次取平均"""
    model.eval()
    # 预热：排除初始化开销
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)
        torch.cuda.synchronize()  # GPU同步

    # 正式测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_data)
        torch.cuda.synchronize()  # GPU同步
    total_time = time.time() - start_time

    # 计算单次推理时间（ms）
    avg_time_ms = (total_time / repeat) * 1000
    return avg_time_ms

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights, weights_only=False)
    model.load_state_dict(state_dict["model"])

    with torch.no_grad():
        # --------- 开始计时 ---------
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        model.eval()
        # prediction
        params_m, total_params = count_model_parameters(model)
        print(f"\n===== 模型参数量 =====")
        print(f"总参数量: {total_params:,} 个 = {params_m:.2f} M")

        data_dict = to_cuda(data_dict)
        data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                  cfg.backbone.num_stages,
                                  cfg.backbone.num_neighbors)
        data_dict.update(data)
        # 3. 计算FLOPs（G）：注意PAREConv输入是dict，需转换为tuple传入profile
        print(f"\n===== 计算FLOPs（请稍等） =====")
        flops, _ = profile(model, inputs=(data_dict,), verbose=False)
        flops_g = flops / 1e9  # 转换为G（十亿）
        print(f"模型总FLOPs: {flops_g:.2f} G")

        # 4. 计算推理时间（ms）
        print(f"\n===== 计算推理时间（请稍等） =====")
        inference_time_ms = calculate_inference_latency(model, data_dict)
        print(f"平均推理时间: {inference_time_ms:.2f} ms/次")
        # ====================== 新增结束 ======================

        output_dict = model(data_dict)
        # --------- 结束计时 ---------
        torch.cuda.synchronize()
        end_ns = time.perf_counter_ns()

        data_dict = release_cuda(data_dict)
        output_dict = release_cuda(output_dict)

    if args.vis_patches:
        visualize_patches(output_dict, args)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]


    # visualization
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    # draw_geometries(ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform)
    # draw_geometries(ref_pcd, src_pcd)

    # compute error
    if args.gt_file is not None:
        transform = data_dict["transform"]
        rre, rte = compute_registration_error(transform, estimated_transform)
        print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
