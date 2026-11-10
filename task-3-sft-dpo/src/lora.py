import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha):
        super().__init__()
        if not isinstance(r, int) or r <= 0:
            raise ValueError("LoRA秩必须为正整数")
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(base.weight.new_zeros(r, in_f))
        self.B = nn.Parameter(base.weight.new_zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def inject_lora(model: nn.modules, target_modules: list[str], r, alpha):
    """原地注入 lora, 返回 model"""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("模型已经含有 LoRA, 请从未注入的副本开始")
    matches = [(name, m) for name, m in model.named_modules()
               if (isinstance(m, nn.Linear) and name.split('.')[-1] in target_modules)]
    if not matches:
        raise ValueError("没有匹配的线性层, 请核对模块名称.")
    model.requires_grad_(False)
    for name, module in matches:
        parent = model
        *path, leaf = name.split('.')
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'可训练 {n_train:,} / 总 {n_total:,} ({n_train / n_total * 100 :.2f}%)')
    return model

def merge_lora(model: nn.modules):
    """把 lora 的增量合入原权重, 返回 model"""
    if not any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("模型不含 LoRA")
    matches = [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    with torch.no_grad():
        for name, module in matches:
            merged = module.base
            merged.weight += (module.B @ module.A) * module.scaling
            parent = model
            *path, leaf = name.split('.')
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, leaf, merged)
    return model

def save_lora(model, path):
    """保存 lora 参数和重建注入结构所需要的配置"""


def load_lora(model, path):
    """根据配置向基座注入 lora 并恢复参数, 返回 model"""

if __name__ == '__main__':
    # ROOT = Path(__file__).resolve().parents[1]
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(ROOT/"models"/"Qwen2.5-0.5B")
    # inject_lora(model, target_modules=['q_proj', 'v_proj'], r=8, alpha=16)

    ...