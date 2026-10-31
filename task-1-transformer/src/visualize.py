import torch
from pathlib import Path
from src.model import load_for_eval
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
ckpt = ROOT / "ckpt" / "best.pt"

model, tokenize = load_for_eval(str(ckpt))

attn_weights = []
def hook_fn(module, input, output):
    attn_weights.append(module.attn_weight.detach().cpu())

handles = []
for block in model.backbone.blocks:
    handles.append(block.attn.register_forward_hook(hook_fn))

# x = tokenize("这家酒店服务很热情，房间干净整洁。").unsqueeze(0)
# seq = "地理位置很方便，出门就是地铁站"
seq = "设施陈旧、卫生太差、体验极糟"
x = tokenize(seq).unsqueeze(0)
logits = model(x)
y = torch.argmax(logits, dim=-1).item()
print(y)

for h in handles:
    h.remove()

print(x[0][:len(seq)])

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
plt.title("Attention Heatmap")
for h in range(4):
    sns.heatmap(
        attn_weights[-1][0][h][:len(seq), :len(seq)].numpy(),
        ax=axes[h],
        xticklabels=list(seq),
        yticklabels=list(seq),
        cmap="magma",
        cbar=False
    )
    axes[h].set_title(f"Head {h + 1}")
    axes[h].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "test.png")
plt.close()