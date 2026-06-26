import torch
import torch.nn as nn

from pareconv.modules.transformer.lrpe_transformer import LRPETransformerLayer
from pareconv.modules.transformer.pe_transformer import PETransformerLayer
from pareconv.modules.transformer.rpe_transformer import RPETransformerLayer
from pareconv.modules.transformer.vanilla_transformer import TransformerLayer
from pareconv.modules.transformer.bias_transformer import  BiasTransformerLayer
from einops import rearrange

# NOTE: SparseTransformerLayer (cast.spot_attention) 和 pytorch3d 仅被下面的 Extended/Spot
# 注意力模块使用（KITTI 的 ExtendedGeometricTransformer 路径）。改为惰性导入，使不依赖它们的
# RPEConditionalTransformer（3DMatch 的 GeometricTransformer 路径）无需安装 pytorch3d 即可运行。
def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,  # 指定Transformer块类型序列，如['self','cross','self','cross']
        d_model,  # 特征维度
        num_heads, # 注意力头数
        dropout=None, # Dropout概率（可选）
        activation_fn='ReLU', # 激活函数类型（默认为ReLU）
        return_attention_scores=False, # 是否返回注意力分数
        parallel=False, # 是否并行处理交叉注意力
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks # 存储块序列
        layers = []
        for block in self.blocks:
            _check_block_type(block)  # 验证block类型有效性
            if block == 'self': # 自注意力层使用相对位置编码的Transformer
                # layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))

            else: # 交叉注意力层使用标准Transformer
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers) # 将层序列转换为ModuleList
        self.return_attention_scores = return_attention_scores   # 配置参数
        self.parallel = parallel

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        """
                前向传播处理两组特征

                参数:
                feats0: 第一组点特征 (B, N, C)
                feats1: 第二组点特征 (B, M, C)
                embeddings0: 第一组点位置编码 (B, N, D)
                embeddings1: 第二组点位置编码 (B, M, D)
                masks0: 第一组点掩码 (B, N) - 可选
                masks1: 第二组点掩码 (B, M) - 可选

                返回:
                增强后的特征（以及可选的注意力分数）
        """
        attention_scores = [] # 存储各层注意力分数
        for i, block in enumerate(self.blocks): # 逐层处理每个Transformer块
            if block == 'self':  # 自注意力块：分别增强各组内部特征
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else: # 交叉注意力块
                if self.parallel:  # 并行处理交叉注意力（同时计算两个方向）
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else: # 串行处理交叉注意力（顺序处理两个方向）
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores: # 如果需要记录注意力分数
                attention_scores.append([scores0, scores1])
            # if torch.isnan(feats0).any() or torch.isnan(feats1).any():
            #
            #     print(feats0)
            #     print(feats1)
            #     print(i, block)
            #     pdb.set_trace()
        if self.return_attention_scores: # 根据配置返回相应结果
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class ConsistencyAwareSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super().__init__()
        from pareconv.modules.cast.spot_attention import SparseTransformerLayer
        # 一致性感知自注意力层
        self.attention = SparseTransformerLayer(
            d_model, num_heads, pe=True,
            dropout=dropout, activation_fn=activation_fn
        )

    def forward(self, feats, indices, attention_mask=None):
        """
        feats: 点特征 (B, N, C)
        indices: 种子点索引 (B, N, K)
        """
        # 扩展种子点索引以匹配注意力层输入格式
        #expanded_indices = indices.expand(-1, feats.size(1), -1)
        return self.attention(feats, feats, indices, attention_mask=attention_mask)


class SpotGuidedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, spots, spot_k, dropout=None, activation_fn='ReLU'):
        super().__init__()
        from pareconv.modules.cast.spot_attention import SparseTransformerLayer
        self.spots = spots
        self.spot_k = spot_k
        # Spot引导交叉注意力层
        self.attention = SparseTransformerLayer(
            d_model, num_heads, pe=False,
            dropout=dropout, activation_fn=activation_fn
        )

    @torch.no_grad()
    def select_spots(self, input_knn, memory_knn, confidence_scores, matching_indices):
        """动态选择关键区域(spots)"""
        from pytorch3d.ops import knn_gather
        # 1. 收集邻居点的置信度分数
        knn_scores = knn_gather(confidence_scores, input_knn[..., 1:]).squeeze(-1)
        # 2. 选择置信度最高的spots个邻居点
        confidence_scores, confident_knn = knn_scores.topk(k=self.spots)
        # 3. 获取高置信邻居点的索引
        confident_knn = torch.gather(input_knn[..., 1:], -1, confident_knn)
        # 4. 添加输入点自身到spot区域
        confident_knn = torch.cat([input_knn[..., :1], confident_knn], dim=-1)
        # 5. 找到每个spot对应的匹配点
        spot_indices = knn_gather(matching_indices, confident_knn).squeeze(-1)
        # 6. 收集匹配点的邻居作为spot区域
        spot_indices = knn_gather(memory_knn, spot_indices)
        # 7. 展平spot区域
        spot_indices = rearrange(spot_indices, 'b n s k -> b n (s k)')
        # 8. 创建注意力掩码
        B, N, M = input_knn.shape[0], input_knn.shape[1], memory_knn.shape[1]
        attention_mask = torch.zeros((B, N, M), device=input_knn.device)
        attention_mask.scatter_(-1, spot_indices, 1.)
        # 9. 获取排序后的稀疏注意力掩码和索引
        # print("attention_mask:",attention_mask.shape)
        spot_mask, spot_indices = attention_mask.topk(spot_indices.shape[-1])
        return spot_mask, spot_indices

    def forward(self, query_feats, memory_feats, input_knn, memory_knn, confidence_scores, matching_indices):
        """
        query_feats: 查询点特征 (B, N, C)
        memory_feats: 内存点特征 (B, M, C)
        input_knn: 输入点K近邻 (B, N, K)
        memory_knn: 内存点K近邻 (B, M, K)
        confidence_scores: 匹配置信度 (B, N, 1)
        matching_indices: 匹配点索引 (B, N, 1)
        """
        # 动态选择spots区域
        spot_mask, spot_indices = self.select_spots(
            input_knn, memory_knn, confidence_scores, matching_indices
        )

        # 应用Spot引导的交叉注意力
        return self.attention(
            query_feats,
            memory_feats,
            spot_indices,
            attention_mask=spot_mask
        )


class ExtendedConditionalTransformer(nn.Module):
    def __init__(
            self,
            blocks1,
            hidden_dim,
            num_heads,
            spots,
            spot_k,
            sigma_c,
            seed_threshold,
            seed_num,
            dual_normalization,
            dropout=None,
            activation_fn='Relu',
            return_attention_scores=True,
            parallel=False
    ):
        super(ExtendedConditionalTransformer, self).__init__()
        # 保存新参数
        self.spots = spots
        self.spot_k = spot_k
        self.sigma_c = sigma_c
        self.seed_threshold = seed_threshold
        self.seed_num = seed_num
        self.dual_normalization = dual_normalization
        self.blocks = blocks1
        # 替换层序列
        self.layers = nn.ModuleList()
        self.return_attention_scores = return_attention_scores  # 配置参数
        self.parallel = parallel
        self.spot_guided_attentions = nn.ModuleList()
        self.consistency_aware_attentions = nn.ModuleList()
        for block in self.blocks:
            if block == 'self':
                self.layers.append(RPETransformerLayer(hidden_dim, num_heads, dropout, activation_fn))
            elif block == 'cross':
                self.layers.append(TransformerLayer(hidden_dim, num_heads, dropout, activation_fn))
            elif block == 'caa-sga':  # 新模块：一致性感知自注意力（需要 pytorch3d + SparseTransformerLayer）
                from pareconv.modules.cast.spot_attention import SparseTransformerLayer  # 惰性导入
                self.layers.append(None)
                self.consistency_aware_attentions.append(SparseTransformerLayer(
                    hidden_dim, num_heads, True, dropout, activation_fn
                ))
                self.spot_guided_attentions.append(SparseTransformerLayer(
                    hidden_dim, num_heads, False, dropout, activation_fn
                ))

            else:
                raise ValueError(f"未知的块类型: {block}")

    def matching_scores(self, input_states: torch.Tensor, memory_states: torch.Tensor):
        if input_states.ndim == 2:
            matching_scores = torch.einsum('mc,nc->mn', input_states, memory_states)
        else:
            matching_scores = torch.einsum('bmc,bnc->bmn', input_states, memory_states)
        if self.dual_normalization:
            ref_matching_scores = torch.softmax(matching_scores, dim=-1)
            src_matching_scores = torch.softmax(matching_scores, dim=-2)
            matching_scores = ref_matching_scores * src_matching_scores
        return matching_scores

    @torch.no_grad()
    def compatibility_scores(self, ref_dists, src_dists, matching_indices):
        """
        Args:
            ref_dists (Tensor): (B, N, N)
            src_dists (Tensor): (B, M, M)
            matching_indices (Tensor): (B, N, 1)

        Returns:
            compatibility (Tensor): (B, N, N)
        """
        from pytorch3d.ops import knn_gather
        #print(f"src_dists:{src_dists.shape}, matching_indices:{matching_indices.shape}")
        src_dists = knn_gather(src_dists, matching_indices).squeeze(2)  # (B, N, 1, M)
        src_dists = knn_gather(src_dists.transpose(1, 2), matching_indices).squeeze(2)  # (B, N, N)
        return torch.relu(1. - torch.abs(ref_dists - src_dists) / self.sigma_c)  # (B, N, N)

    @torch.no_grad()
    def seeding(self, compatible_scores: torch.Tensor, confidence_scores: torch.Tensor):
        selection_scores = compatible_scores.lt(compatible_scores.max(-1, True)[0] * self.seed_threshold)  # (B, N)
        max_num = torch.clamp_max(selection_scores.gt(0).int().sum(-1).min(), self.seed_num)
        selection_scores = selection_scores.float() * confidence_scores.squeeze(-1)  # (B, N)
        return selection_scores.topk(max_num, dim=-1).indices  # (B, K)

    def forward(
        self,
        feats0,
        feats1,
        embeddings0,
        embeddings1,
        dists0,  # 新增：参考点距离矩阵 (B, N, N)
        dists1,  # 新增：源点距离矩阵 (B, M, M)
        idx0,  # 新增：参考点K近邻索引 (B, N, K)
        idx1,  # 新增：源点K近邻索引 (B, M, K)
        masks0=None,
        masks1=None,
    ):
        attention_scores = []
        correlation = []
        ref_compatibility = []
        src_compatibility = []
        j = 0
        # print(self.blocks)
        for i, block in enumerate(self.blocks):
            if block == 'self':
                # 原始自注意力
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)

            elif block == 'cross':
                # 原始交叉注意力
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)

            # elif block == 'consistency_self':
            #     # 计算匹配分数和兼容性
            #     matching_scores = self.matching_scores(feats0, feats1)
            #     confidence0, matching_idx0 = torch.max(matching_scores, dim=-1, keepdim=True)
            #     compatibility0 = self.compatibility_scores(dists0, dists1, matching_idx0).mean(-1)
            #     confidence0 = confidence0 * compatibility0.unsqueeze(-1)
            #     # print("token_indices0:", confidence0.shape)
            #     print("compatibility0:", compatibility0.shape)
            #     # 选择种子点
            #     token_indices0 = self.seeding(compatibility0, confidence0)
            #     #print("token_indices0:", token_indices0.shape)
            #     # 应用一致性感知自注意力
            #     feats0 = self.layers[i](feats0, token_indices0.unsqueeze(1))
            #     feats1 = self.layers[i](feats1, token_indices0.unsqueeze(1))
            #     scores0 = scores1 = None  # 此模块不返回传统注意力分数

            elif block == 'caa-sga':
                # 计算匹配分数
                # matching_scores = self.matching_scores(feats0, feats1)
                # confidence0, matching_idx0 = torch.max(matching_scores, dim=-1, keepdim=True)

                matching_scores = self.matching_scores(feats0, feats1)
                #print(f" matching_scores 1:{matching_scores.shape}")
                correlation.append(matching_scores)

                confidence_scores, matching_indices = torch.max(matching_scores, dim=-1, keepdim=True)
                #print(f" confidence_scores0:{confidence_scores.shape},matching_indices0:{matching_indices.shape}")
                compatible_scores = self.compatibility_scores(dists0, dists1, matching_indices).mean(-1)
                confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
                ref_token_indices = self.seeding(compatible_scores, confidence_scores)
                # 应用Spot引导交叉注意力（select_spots 的 self.spots 已在类内部使用，不需要再传）
                ref_spot_mask, ref_spot_indices = self.spot_guided_attentions[j].select_spots(
                    idx0[..., :self.spot_k + 1],
                    idx1[..., :self.spot_k],
                    confidence_scores,
                    matching_indices,
                )
                ref_compatibility.append(compatible_scores)

                # 对称处理另一组点
                confidence_scores, matching_indices = torch.max(matching_scores.transpose(1, 2), dim=-1, keepdim=True)
                #print(f" confidence_scores1:{confidence_scores.shape},matching_indices1:{matching_indices.shape}")
                compatible_scores = self.compatibility_scores(dists1, dists0, matching_indices).mean(-1)
                confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
                src_token_indices = self.seeding(compatible_scores, confidence_scores)
                src_spot_mask, src_spot_indices = self.spot_guided_attentions[j].select_spots(
                    idx1[..., :self.spot_k + 1],
                    idx0[..., :self.spot_k],
                    confidence_scores,
                    matching_indices,
                )
                src_compatibility.append(compatible_scores)

                feats0 = self.consistency_aware_attentions[j](
                    feats0, feats0, ref_token_indices.unsqueeze(1)
                )
                feats1 = self.consistency_aware_attentions[j](
                    feats1, feats1, src_token_indices.unsqueeze(1)
                )
                feats0 = self.spot_guided_attentions[j](
                    feats0, feats1, ref_spot_indices, attention_mask=ref_spot_mask
                )
                feats1 = self.spot_guided_attentions[j](
                    feats1, feats0, src_spot_indices, attention_mask=src_spot_mask
                )
                j += 1

                scores0 = torch.stack(ref_compatibility, dim=-1)
                scores1 = torch.stack(src_compatibility, dim=-1)
            if self.return_attention_scores and scores0 is not None:
                attention_scores.append([scores0, scores1])

        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
class BiasConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(BiasConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                # layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers.append(BiasTransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))

            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1