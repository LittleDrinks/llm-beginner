import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor|None=None) -> torch.tensor:
    """
    输入：
    Q: (B, H, T, D)
    K: (B, H, T, D)
    K: (B, H, T, D)

    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V

    输出：
    output: (B, H, T, D)
    """
    d_k = Q.shape[-1]
    score = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        assert(mask.dtype == torch.bool)
        score = score.masked_fill(mask, float('-inf'))
    attention = torch.softmax(score, dim=-1)
    return attention @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        if d_model % num_heads != 0:
            raise ValueError("Invalid head num!")
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, mask: torch.Tensor|None = None) -> torch.Tensor:
        """
        输入：
        X: (B, T, D) 若干句句子

        PIPELINE:
        X: (B, T, D)
        X @ W_{Q/K/V}:(B, T, D) -> Q, K, V: (B, T, D)
        reshape -> multi head: (B, H, T, d_k)
        scaled_dot_product_attention() -> (B, H, T, d_k)
        -> 多头合并
        -> O: (B, T, D)

        输出：
        O: (B, T, D)
        """
        B, T, _ = X.shape
        Q = self.W_Q(X).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        O = scaled_dot_product_attention(Q, K, V, mask)
        O = O.transpose(1, 2).reshape(B, T, self.d_model)
        O = self.W_O(O)
        return O
