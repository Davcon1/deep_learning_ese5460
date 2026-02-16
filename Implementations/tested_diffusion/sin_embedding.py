import torch
import torch.nn as nn
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dims: int = 512):
        super().__init__()
        self.embedding_dims = embedding_dims

    def forward(self, t) :
        device = t.device
        k = self.embedding_dims // 2
        vals = torch.arange(k, device=device, dtype=torch.float32)
        # generate the embeddings
        emb = torch.exp(-math.log(10000.0) * vals / k)
        angles = t.unsqueeze(-1).float() * emb

        # store the sin and cosines in a single tensor
        pe = torch.zeros(*t.shape, self.embedding_dims, device=device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return pe
