## step1 

先实际调用一下需要实现的函数

```py
import torch
import torch.nn.functional as F

torch.manual_seed(42)

B = 1  # batch size
H = 1  # head
T = 3  # sequence length
D = 4  # token embedding 的 维度


Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

print('Q.shape : ', Q.shape)
print('K.shape : ', K.shape)
print('V.shape : ', V.shape)

out = F.scaled_dot_product_attention(Q, K, V)
print("out.shape : ", out.shape)
```

输出：

```
Q.shape :  torch.Size([1, 1, 3, 4])
K.shape :  torch.Size([1, 1, 3, 4])
V.shape :  torch.Size([1, 1, 3, 4])
out.shape :  torch.Size([1, 1, 3, 4])
```

需要实现的大概是让一行 tokens 吸收上下文的内容

Attention 的核心公式是

$$
Attention(Q,K,V) = softmax\bigg(\frac{QK^T}{\sqrt{d_k}}\bigg)V
$$

$(T,D)$ 的矩阵 $Q$ 和 $(D,T)$ 的矩阵 $K^T$ 相乘得到一个 $(T,T)$ 的矩阵，表示第 $i$ 个 token 和第 $j$ 个 token 的关系

可以认为 $Q,K$ 每个位置都是 0~1 的随机变量，均值 $0$，方差 $1$，可以认为 $Var(s_{i,j})=d_k$，于是为了归一化除以 $\sqrt{d_k}$，再过一层 softmax 转化为概率分布。

在此基础上，$(T,T)\times (T,D)$ 可以得到一个 $(T,D)$ 的矩阵，每一行加权平均用于吸收其他的 token

到这里就可以实现我们需要的 `scaled_dot_product_attention`

```py
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
    
```
