import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor) -> torch.tensor:
    """
    输入：
    Q: (B, H, T, D)
    K: (B, H, T, D)
    K: (B, H, T, D)

    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V

    输出：
    output: (B, H, T, D)
    """
    (_, _, _, D) = Q.shape
    return torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim=-1) @ V
    