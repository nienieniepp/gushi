import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[“”‘’]", "\"", text)
    text = re.sub(r"[（）]", "", text)
    return text.strip()


def load_poems_from_json(json_path: Path) -> List[str]:
    poems: List[str] = []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            paragraphs = item.get("paragraphs", [])
            if paragraphs:
                poems.append("".join(paragraphs))
    return poems


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chinese-poetry data for GPT-2 training")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=160)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_poems: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".json"):
                all_poems.extend(load_poems_from_json(Path(root) / name))

    cleaned: List[str] = []
    for poem in all_poems:
        poem = clean_text(poem)
        if args.min_len <= len(poem) <= args.max_len:
            cleaned.append(poem)

    cleaned = sorted(set(cleaned))

    with output_file.open("w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")

    stats: Dict[str, float] = {
        "num_raw_poems": len(all_poems),
        "num_cleaned_poems": len(cleaned),
        "avg_length": round(sum(len(x) for x in cleaned) / max(len(cleaned), 1), 3),
        "min_len": args.min_len,
        "max_len": args.max_len,
    }

    stats_file = output_file.parent / "stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned data to: {output_file}")
    print(f"Saved stats to: {stats_file}")


if __name__ == "__main__":
    main()
