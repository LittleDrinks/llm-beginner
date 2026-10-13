import torch
import torch.nn as nn
from tqdm import tqdm
from src.model import TransformerClassifier
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CKPT_DIR = Path(__file__).resolve().parents[1] / "ckpt"


def encode(text, vocab, max_len=512):
    ids = [vocab.get(ch, vocab["<unk>"]) for ch in text]
    ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return ids


def main():
    # data
    train_data = pd.read_parquet(DATA_DIR / "train.parquet")
    val_data = pd.read_parquet(DATA_DIR / "validation.parquet")
    # print("==================== preview ====================")
    # print(f"样本数: {len(train_data)}")
    # print(f"列名: {train_data.columns.to_list()}")
    # print(f"第一条样本: {train_data.iloc[0].to_dict()}")

    # 建词表
    vocab = {"<pad>": 0, "<unk>": 1}
    for char in sorted(
        {char for text in train_data["text"].to_list() for char in text}
    ):
        vocab[char] = len(vocab)

    # 建数据集
    train_set = TensorDataset(
        torch.tensor(
            [encode(text, vocab) for text in train_data["text"]], dtype=torch.long
        ),
        torch.tensor(train_data["label"].to_list(), dtype=torch.long),
    )
    val_set = TensorDataset(
        torch.tensor(
            [encode(text, vocab) for text in val_data["text"]], dtype=torch.long
        ),
        torch.tensor(val_data["label"].to_list(), dtype=torch.long),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model_config = {
        "vocab_size": len(vocab),
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "pad_id": vocab["<pad>"],
        "num_classes": 2,
        "max_len": 512,
    }
    model = TransformerClassifier(**model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = 7
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_config": model_config,
        "vocab": vocab,
    }
    (CKPT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        tot_loss = 0
        loop = tqdm(train_loader, desc=f"epoch[{epoch + 1}/{epochs}]")
        for X, Y in loop:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = criterion(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        print(
            f"epoch[{epoch + 1}/{epochs}] avg loss = {tot_loss / len(train_loader):.4f}"
        )

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                correct += (pred == Y).sum().item()
                total += Y.size(0)
        val_acc = correct / total
        print(f"epoch[{epoch + 1}/{epochs}] val acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            print(f"已保存最佳模型, val acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
