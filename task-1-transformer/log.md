## test 1

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

可以认为 $Q,K$ 每个位置都是随机变量，均值 $0$，方差 $1$，可以认为 $Var(s_{i,j})=d_k$，于是为了归一化除以 $\sqrt{d_k}$，再过一层 softmax 转化为概率分布。

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

## test 2

在 test1 的基础上，需要实现 mask

一种 mask 是 causal mask，因果矩阵，在拿到 $QK^T$ 这个 $(T,T)$ 的矩阵之后，需要屏蔽一些【这个 token 不能看到的 token】，比如“我 喜欢 西瓜”，预测下一个 token 时不能看下一个

另一种 mask 是 padding mask，为了把所有的语句长度都统一成 T，需要在句子后面加上若干个 <pad>，此时其他 token 需要屏蔽这些 <pad>

我们需要在 Attention 函数中传入 mask，其中 mask 是一个 $(T,T)$ 的矩阵，True 表示该位置被屏蔽

```py
def scaled_dot_product_attention(Q, K, V, mask)
```

为了让 softmax 能够将被屏蔽的位置输出 0，应当将输入设为 -inf

```py
score = Q @ K.transpose(-2, -1)
score = score.masked_fill(mask, float('-inf'))
```
