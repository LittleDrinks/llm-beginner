import torch
import torch.nn as nn
from src.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.FFN = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.LN1 = LayerNorm(d_model)
        self.LN2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.LN1(x + self.attn(x, mask=mask))
        x = self.LN2(x + self.FFN(x))
        return x
