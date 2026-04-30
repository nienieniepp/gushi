#!/usr/bin/env python3
"""
处理 chinese-poetry（poet.tang.*.json）数据集的脚本。

功能：
1. 遍历输入目录中的全部 poet.tang.*.json 文件（不会只读单文件）。
2. 筛选仅保留：
   - 恰好 4 句
   - 每句恰好 5 或 7 个中文字符
3. 将四句使用“。”拼接为一行。
4. 去重后写入文本文件（每行一首诗）。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set


# 仅保留中文汉字，用于严格统计“中文字符数”
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def count_chinese_chars(text: str) -> int:
    """统计字符串中的中文汉字数量。"""
    return len(CHINESE_CHAR_RE.findall(text))


def is_valid_poem(paragraphs: List[str]) -> bool:
    """判断是否满足：4句，每句5或7个中文字符。"""
    if len(paragraphs) != 4:
        return False

    for line in paragraphs:
        clean_line = line.strip()
        # 这里按“中文字符数”判断，忽略标点和空白
        n = count_chinese_chars(clean_line)
        if n not in (5, 7):
            return False
    return True


def iter_tang_json_files(input_dir: Path) -> Iterable[Path]:
    """递归遍历目录，返回所有 poet.tang.*.json 文件。"""
    # 使用 rglob 递归扫描，确保处理目录下所有匹配文件
    for p in sorted(input_dir.rglob("poet.tang.*.json")):
        if p.is_file():
            yield p


def load_poems_from_file(json_file: Path) -> List[str]:
    """从单个 JSON 文件中提取满足条件的诗并拼接。"""
    poems: List[str] = []

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return poems

    for item in data:
        paragraphs = item.get("paragraphs", [])
        if isinstance(paragraphs, list) and is_valid_poem(paragraphs):
            # 每句去掉首尾空格后，用“。”连接
            normalized = [line.strip() for line in paragraphs]
            poem = "。".join(normalized)
            poems.append(poem)

    return poems


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 chinese-poetry 唐诗数据")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 poet.tang.*.json 的目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出 txt 文件（每行一首）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_poems: List[str] = []
    file_count = 0

    # 遍历并处理全部匹配的 JSON 文件
    for json_file in iter_tang_json_files(input_dir):
        file_count += 1
        all_poems.extend(load_poems_from_file(json_file))

    # 去重并排序（便于复现实验）
    unique_poems: Set[str] = set(all_poems)
    sorted_poems = sorted(unique_poems)

    with output_file.open("w", encoding="utf-8") as f:
        for poem in sorted_poems:
            f.write(poem + "\n")

    print(f"扫描文件数: {file_count}")
    print(f"原始符合条件诗数: {len(all_poems)}")
    print(f"去重后诗数: {len(sorted_poems)}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()
