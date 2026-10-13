import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from src.model import TransformerBackbone
from tqdm import tqdm


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        pad_id: int = 0,
        eos_id: int = 1,
        max_len: int = 512,
    ):
        super().__init__()
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.backbone = TransformerBackbone(
            vocab_size, d_model, num_heads, num_layers, pad_id, max_len
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # 自动根据当前输入序列长度 T 构造 Causal Mask: (T, T) 上三角为 True
        # 透传给 backbone
        # logits = self.lm_head(x) -> (B, T, vocab_size)
        B, T = ids.shape

        # 因果遮罩
        causal_mask = torch.triu(
            torch.ones(T, T, device=ids.device, dtype=torch.bool), diagonal=1
        )

        # padding mask
        padding_mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(2)

        h = self.backbone(ids, causal_mask | padding_mask)
        logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # 自回归采样循环：逐步预测、追加、再喂入
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            if next_token == self.eos_id:
                break
        return ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"toy LM runs on {device}")

    # 取出诗句
    POEM_PATH = Path(__file__).resolve().parent.parent.parent / "poetryFromTang.txt"
    raw = open(POEM_PATH).read().replace("\r\n", "\n")
    poem = [p.replace("\n", "").strip() for p in raw.split("\n\n")]

    # 构建词表
    chars = sorted(list(set(raw)) + ["<pad>", "<eos>"])
    vocab_size = len(chars)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    print(f"词表大小总数 {vocab_size}")

    # 构建输入和目标
    data = [
        torch.tensor([char2id[c] for c in p] + [char2id["<eos>"]], device=device)
        for p in poem
    ]

    def collate_fn(batch):
        # batch (B,...) -> (B,T)
        padded = nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=char2id["<pad>"]
        )
        return padded[:, :-1], padded[:, 1:]

    loader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate_fn)

    #
    model = TransformerLM(
        vocab_size, 128, 4, 4, char2id["<pad>"], char2id["<eos>"], 65536
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=char2id["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    model.train()
    for epoch in range(50):
        loop = tqdm(loader)
        total_loss = 0
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            logits = model(x)  # (B, T, vocab_size)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        if epoch % 10 == 9:
            print(
                f"[epoch {epoch + 1:02d}/50] avg_loss = {loss.item() / len(loader):.4f}"
            )

    # eval
    model.eval()
    prompt = "床"
    prompt_ids = torch.tensor([[char2id[x] for x in prompt]], device=device)
    out_ids = model.generate(prompt_ids, 25)  # (B, T)
    generated_text = "".join([id2char[x.item()] for x in out_ids[0]])
    print("--------------- 自回归测试 ---------------")
    print(f"prompt: {prompt}")
    print(f"generate: {generated_text}")
    print("-----------------------------------------")
