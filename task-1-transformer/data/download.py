"""下载 ChnSentiCorp 中文情感分类数据集到当前目录。

依赖：pip install datasets pyarrow

默认从 Hugging Face 下载。如果国内访问 HF 不稳定：
  - 方案一（推荐）：设置环境变量 HF_ENDPOINT=https://hf-mirror.com
  - 方案二：用 ModelScope（见脚本末尾提示）
"""

"""下载 ChnSentiCorp 数据集到当前目录。"""

from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download


DATA_DIR = Path(__file__).parent

FILES = {
    "train": "chn_senti_corp-train.arrow",
    "validation": "chn_senti_corp-validation.arrow",
    "test": "chn_senti_corp-test.arrow",
}


def main():
    print("正在下载 seamew/ChnSentiCorp ...")

    for split, filename in FILES.items():
        print(f"正在下载 {split} ...")

        path = hf_hub_download(
            repo_id="seamew/ChnSentiCorp",
            filename=filename,
            repo_type="dataset",
            cache_dir=str(DATA_DIR / "cache"),
        )

        ds = Dataset.from_file(path)

        out = DATA_DIR / f"{split}.parquet"
        ds.to_parquet(str(out))

        print(f"  {split}: {len(ds)} 条 -> {out.name}")

    print(f"\n完成。数据保存在 {DATA_DIR}")


if __name__ == "__main__":
    main()