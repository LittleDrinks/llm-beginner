import json
from itertools import islice
from pathlib import Path

from tqdm import tqdm

DATA_ROOT = Path(__file__).parents[1] / "data" / "moss-sft"


def to_message(record: dict):
    messages = []
    messages.append({"role": "system", "content": record["meta_instruction"]})
    for i in range(record["num_turns"]):
        messages.append(
            {
                "role": "user",
                "content": record["chat"][f"turn_{i + 1}"]["Human"]
                .removeprefix("<|Human|>: ")
                .removesuffix("<eoh>\n"),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": record["chat"][f"turn_{i + 1}"]["MOSS"]
                .removeprefix("<|MOSS|>: ")
                .removesuffix("<eom>\n"),
            }
        )
    return messages


def convert_jsonl(src: Path, dst: Path, limit: int | None = None) -> None:
    if limit <= 0:
        raise ValueError('limit <= 0!')
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        lines = f if limit is None else islice(f, limit)
        for line in tqdm(lines, total=limit, unit="条"):
            record = json.loads(line)
            sample = {
                "conversation_id": record["conversation_id"],
                "messages": to_message(record),
            }
            out.write(json.dumps(sample, ensure_ascii=False))
            out.write("\n")


if __name__ == "__main__":
    src = DATA_ROOT / "moss-003-sft-no-tools.jsonl"
    dst = DATA_ROOT / "sft.jsonl"
    convert_jsonl(src, dst, 5000)
