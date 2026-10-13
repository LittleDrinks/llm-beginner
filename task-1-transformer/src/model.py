import torch
import torch.nn as nn
from src.block import TransformerBlock
import json
from pathlib import Path

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        pad_id: int = 0,
        num_classes: int = 2,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.num_classes = num_classes

        # 1. token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 2. Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        # 3. APE
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = 10000.0 ** (
            -torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        angles = position * frequency
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])  # 支持 d_model 是奇数的情况
        self.register_buffer("position_encoding", pe.unsqueeze(0))

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, T)
        B, T = ids.shape

        if T > self.position_encoding.shape[1]:
            raise ValueError(
                f"sequence length {T} exceed max_len {self.position_encoding.shape[1]}"
            )

        if (ids == self.pad_id).all(dim=1).any():
            raise ValueError(f"all tokens are <pad>!")

        # padding mask: (B, 1, 1, T)
        # 将所有输入的长度统一为 T, 屏蔽那些用于占位的 <pad>
        # ids == self.pad_id 得到一个 (B, T) 的布尔矩阵, True 表示该位置需要屏蔽
        # mask 是作用在 score=QK^T/sqrt(d_k) 上的
        # score 的形状是 (B, H, T, T), 我们把 mask 的形状设置为 (B, 1, 1, T), 就可以自动广播为 (B, H, T, T), 效果为:
        # ====================================
        #       key→ 我     爱     你    <pad>
        # query↓
        # 我         s00   s01   s02   -inf
        # 爱         s10   s11   s12   -inf
        # 你         s20   s21   s22   -inf
        # <pad>      s30   s31   s32   -inf
        # ====================================
        mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)

        # embedding: (B, T) -> (B, T, d_model)
        x = self.token_embedding(ids)

        # + APE
        x = x + self.position_encoding[:, :T, :]

        # N * Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # masked mean polling: (B, T, D) -> (B, D)
        # 相当于把每个 token 的词向量平均一下, 得到一个新的词向量表达这句话的语义
        # 先把 <pad> 那些位置的词向量置 0, 然后直接加起来, 除以正确的数量即可
        valid = (ids != self.pad_id).unsqueeze(-1)  # (B, T) -> (B, T, 1)
        count = valid.sum(dim=1)
        x = (x * valid).sum(dim=1) / count

        # classify
        logits = self.classifier(x)
        return logits


def load_for_eval(ckpt_path: str):
    """
    model, tokenize_fn = load_for_eval(str(ckpt))
    model.eval()
    """
    ckpt_path = Path(ckpt_path)
    metadata = json.loads((ckpt_path.parent / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["model_config"]
    vocab = metadata["vocab"]
    model = TransformerClassifier(**config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    from src.train import encode

    def tokenize(text: str) -> torch.Tensor:
        return torch.tensor(encode(text, vocab), dtype=torch.long)

    return model, tokenize
