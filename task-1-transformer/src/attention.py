import math
import torch
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

