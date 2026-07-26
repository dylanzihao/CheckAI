"""
PyTorch Dataset：加载 JSONL 数据并使用 BERT Tokenizer 进行编码。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class AITextDataset(Dataset):
    """AI 文本检测数据集，支持 JSONL 格式。"""

    def __init__(
        self,
        data_file: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        self.texts: list[str] = []
        self.labels: list[int] = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.texts.append(record["text"])
                self.labels.append(int(record["label"]))

        logger.info(f"加载数据集: {data_file.name} | 样本数: {len(self)}")

        # 统计标签分布
        human_count = sum(self.labels)
        ai_count = len(self.labels) - human_count
        logger.info(f"  Human: {human_count}, AI: {ai_count}")

        # 统计 token 长度分布（使用 tokenizer 获得准确的 token 数）
        text_lengths = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in self.texts]
        over_max = sum(1 for l in text_lengths if l > max_length * 2)
        pct_no_truncate = sum(1 for l in text_lengths if l <= max_length) / max(len(self.texts), 1) * 100
        logger.info(f"  Token长度范围: {min(text_lengths)} ~ {max(text_lengths)}")
        logger.info(f"  无需截断 (≤{max_length} tokens): {pct_no_truncate:.1f}%")
        if over_max:
            logger.warning(f"  超长文本 (> {max_length*2} tokens): {over_max} 条")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
