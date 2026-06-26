import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.layers import build_dropout_layer


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout=None):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices):
        r"""Learnable Positional Embedding.

        `emb_indices` are truncated to fit the finite embedding space.

        Args:
            emb_indices: torch.LongTensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEmbedding, self).__init__()
        self.linear = nn.Linear(3, d_model // 2)

    def embed(self, emb_coordinates):
        x = self.linear(emb_coordinates)
        return torch.sin(x), torch.cos(x)

    def encode(self, sin_embeddings, cos_embeddings, features):
        feats1 = features[..., 0::2] * cos_embeddings - features[..., 1::2] * sin_embeddings
        feats2 = features[..., 0::2] * sin_embeddings + features[..., 1::2] * cos_embeddings
        return torch.stack([feats1, feats2], dim=-1).view(features.shape)

    def forward(self, emb_coordinates, features):
        sin_embeddings, cos_embeddings = self.embed(emb_coordinates)
        return self.encode(sin_embeddings, cos_embeddings, features)