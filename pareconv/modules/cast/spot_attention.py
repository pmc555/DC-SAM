import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
from einops import rearrange # 用于张量维度重组

from pareconv.modules.transformer.output_layer import AttentionOutput
from pareconv.modules.transformer.positional_embedding import RotaryPositionalEmbedding
from pareconv.modules.cast.modules import UnaryBlock


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, query, support, upsample_indices):
        """
        Args:
            query (Tensor): (B, N, C)
            support (Tensor): (B, M, C')
            upsample_indices (Tensor): (B, N, 1)
        return:
            latent (Tensor): (B, N, C)
        """
        latent = knn_gather(support, upsample_indices).squeeze(2)
        return self.output(self.unary(latent) + query)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, q_feats, s_feats, q_points:torch.Tensor, s_points:torch.Tensor, downsample_indices):
        """
        Args:
            q_feats (Tensor): (B, N, C)
            s_feats (Tensor): (B, M, C')
            q_points (Tensor): (B, N, 3)
            s_points (Tensor): (B, N, K, 3)
            downsample_indices (Tensor): (B, N, K)
        return:
            latent (Tensor): (B, M, C)
        """
        grouped_feats = knn_gather(s_feats, downsample_indices) # (B, N, K, C')
        knn_weights = 1. / ((s_points - q_points.unsqueeze(2)).pow(2).sum(-1) + 1e-8) # (B, N, K)
        knn_weights = knn_weights / knn_weights.sum(dim=-1, keepdim=True) # (B, N, K)
        latent = torch.sum(grouped_feats * knn_weights.unsqueeze(-1), dim=2) # (B, N, C)
        return self.output(self.unary(latent) + q_feats)


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, pe=True, dropout=None, activation_fn='relu'):
        """
             稀疏Transformer层初始化

             参数:
             d_model: 特征维度
             num_heads: 注意力头数
             pe: 是否使用位置编码 (默认True)
             dropout: Dropout概率 (可选)
             activation_fn: 激活函数类型 (默认ReLU)
             """
        super(SparseTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads # 每个头的维度
        self.pe = pe # 位置编码标志
        # 线性变换层: 用于生成Q、K、V向量
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        # 输出处理层
        self.linear = nn.Linear(d_model, d_model) # 线性层用于特征融合
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity() # 无dropout
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) # 层归一化
        self.output = AttentionOutput(d_model, dropout, activation_fn)  # 输出处理模块 (可能包含线性层和激活函数)
        # 位置编码模块 (如果启用)
        if pe: self.rpe = RotaryPositionalEmbedding(self.d_model)
    
    @torch.no_grad()  # 此方法不参与梯度计算
    def select_spots(self, input_knn, memory_knn, confidence_scores, matching_indices, num_spots):
        """
        动态选择关键区域(spots)用于稀疏注意力计算
        Args:
            input_knn (Tensor) 输入点的K近邻索引 [包含自身索引]: (B, N, k+1)
            memory_knn (Tensor) 内存点的K近邻索引: (B, M, K)
            confidence_scores (Tensor) 各点的匹配置信度 : (B, N, 1)
            matching_indices (Tensor) 匹配点索引: (B, N, 1)
            num_spots: 要选择的spots数量
        Returns:
            spot_mask: 注意力掩码 (B, N, (S+1)*K)
            spot_indices: 选择的spot区域点索引 (B, N, (S+1)*K)
            output_states: torch.Tensor (B, N, C)
        """
        # 1. 收集邻居点的置信度分数
        knn_scores = knn_gather(confidence_scores, input_knn[...,1:]).squeeze(-1)  # (B, N, k)
        # 2. 选择置信度最高的num_spots个邻居点
        confidence_scores, confident_knn = knn_scores.topk(k=num_spots)  # (B, N, S)
        # 3. 获取高置信邻居点的索引
        confident_knn = torch.gather(input_knn[...,1:], -1, confident_knn)  # (B, N, S)
        # 4. 添加输入点自身到spot区域 (作为中心点)
        confident_knn = torch.cat([input_knn[...,:1], confident_knn], dim=-1)  # (B, N, S+1)
        # 5. 找到每个spot对应的匹配点
        spot_indices = knn_gather(matching_indices, confident_knn).squeeze(-1)  # (B, N, S+1)
        # 6. 收集匹配点的邻居作为spot区域
        spot_indices = knn_gather(memory_knn, spot_indices)  # (B, N, S+1, K)
        # 7. 展平spot区域
        spot_indices = rearrange(spot_indices, 'b n s k -> b n (s k)')  # (B, N, (S+1)*K)
        # 8. 创建注意力掩码 (避免重复计算点)
        # avoid redundant indices from spot areas
        B, N, M = input_knn.shape[0], input_knn.shape[1], memory_knn.shape[1]
        attention_mask = torch.zeros((B, N, M), device=input_knn.device) # 初始化全0掩码
        # 9. 将spot区域点标记为1
        attention_mask.scatter_(-1, spot_indices, 1.)  # (B, N, M)
        # 10. 通过topk获取排序后的稀疏注意力掩码和索引
        spot_mask, spot_indices = attention_mask.topk(spot_indices.shape[-1])  # (B, N, (S+1)*K)
        return spot_mask, spot_indices
    
    def forward(self, input_states, memory_states, indices, input_coord=None, memory_coord=None, attention_mask=None):
        """Sparse Transformer Layer稀疏Transformer前向传播

        Args:
            input_states (Tensor)输入点特征: (B, N, C)
            memory_states (Tensor)内存点特征: (B, M, C)
            indices (Tensor)每个输入点对应的内存点索引: (B, N, K)
            input_coord (Tensor)输入点坐标 [用于位置编码]: (B, N, 3)
            memory_coord (Tensor)内存点坐标[用于位置编码]: (B, M, 3)
            attention_mask (Tensor)注意力掩码: (B, N, K)

        Returns:
            output_states输出特征: torch.Tensor (B, N, C)
        """
        # 确保所有张量在同一设备上
        #indices = indices.to(memory_states.device)

        # 确保索引是整数类型
        # if not indices.dtype.is_floating_point:
        #     indices = indices.long()

        # 获取内存点数量
        #M = memory_states.size(1)  # 211

        # 打印关键信息用于调试
        # print(f"输入点数量: {input_states.size(1)}")
        # print(f"内存点数量: {memory_states}")
        # print(f"索引形状: {indices.shape}")
        # print(f"索引值范围: min={indices.min().item()}, max={indices.max().item()}")
        # print("input_states:", input_states.shape)
        # print("memory_states:", memory_states.shape)
        # print("indices:", indices.shape)
        # 1. 生成查询向量Q
        q = self.proj_q(input_states)  # (B, N, H*C)
        # 2. 生成键向量K - 仅关注indices指定的内存点
        k = knn_gather(self.proj_k(memory_states), indices)  # (B, N, K, H*C)
        # 3. 生成值向量V - 仅关注indices指定的内存点
        v = knn_gather(self.proj_v(memory_states), indices)  # (B, N, K, H*C)
        # 4. 应用旋转位置编码 (如果启用且坐标可用)
        if self.pe and memory_coord is not None and input_coord is not None:
            k = self.rpe(knn_gather(memory_coord, indices) - input_coord.unsqueeze(2), k)
        # 5. 重塑维度用于多头注意力
        q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(k, 'b n m (h c) -> b h n m c', h=self.num_heads)
        v = rearrange(v, 'b n m (h c) -> b h n m c', h=self.num_heads)
        # 6. 计算注意力分数
        # Q·K^T / sqrt(d_k)
        attention_scores = torch.einsum('bhnc,bhnmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        # 7. 应用注意力掩码 (如spot区域掩码)
        if attention_mask is not None:
            # 将不需要关注的位置分数设为极小的负值 (softmax后变为0)
            attention_scores = attention_scores - 1e6 * (1. - attention_mask.unsqueeze(1))
        # 8. 注意力分数归一化
        attention_scores = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.sum(attention_scores.unsqueeze(-1) * v, dim=-2)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        output_states = self.output(output_states)
        return output_states
