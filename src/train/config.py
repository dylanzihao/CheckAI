"""
训练配置：模型超参、路径、训练参数集中管理。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """训练配置（兼容本地和 Kaggle 环境）。"""

    # --- 路径 ---
    # 预训练模型路径（本地 > Kaggle > HuggingFace ID）
    pretrained_model_name: str = "hfl/chinese-roberta-wwm-ext"
    local_model_dir: str = "D:\\Dylan\\Model\\chinese-roberta-wwm-ext"
    kaggle_model_dir: str = "/kaggle/input/models/dylanzihao/chinese-bert-wwm-ext/transformers/default/1"

    # 数据目录：优先使用 Kaggle 输入路径，否则使用项目相对路径
    _project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    kaggle_data_dir: str = "/kaggle/input/datasets/dylanzihao/c-red-processed"
    data_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed",
    ))

    # 输出目录
    output_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "models" / "base",
    ))

    # --- 模型参数 ---
    max_seq_length: int = 512      # BERT 最大输入长度，覆盖 ~63% 数据无需截断
    num_labels: int = 2            # 二分类：Human vs AI
    dropout_rate: float = 0.1      # 分类头 dropout

    # --- 训练超参 ---
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2   # 有效 batch_size = 8 × 2 GPU × 2 = 32
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_ratio: float = 0.1      # linear warmup 比例

    # --- 混合精度 ---
    fp16: bool = True              # T4 支持混合精度加速

    # --- 分布式训练 ---
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", -1)))
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", 1)))

    # --- 日志与保存 ---
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2
    seed: int = 42
    dataloader_num_workers: int = 4

    # --- 其他 ---
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True

    def resolve_model_name(self) -> str:
        """解析模型路径：本地 > Kaggle 输入 > HuggingFace ID。"""
        if Path(self.local_model_dir).exists():
            return self.local_model_dir
        if Path(self.kaggle_model_dir).exists():
            return self.kaggle_model_dir
        return self.pretrained_model_name

    def resolve_data_dir(self) -> str:
        """解析数据目录：Kaggle 输入 > 项目本地路径。"""
        if Path(self.kaggle_data_dir).exists():
            return self.kaggle_data_dir
        return self.data_dir

    @property
    def device(self) -> str:
        """返回当前进程的可用设备。"""
        import torch
        if torch.cuda.is_available():
            return f"cuda:{self.local_rank}" if self.local_rank >= 0 else "cuda"
        return "cpu"

    @property
    def is_distributed(self) -> bool:
        """是否处于分布式训练模式。"""
        return self.local_rank >= 0 and self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """是否为主进程（rank 0）。"""
        return self.local_rank <= 0

    @property
    def effective_batch_size(self) -> int:
        """有效 batch size。"""
        n_gpu = max(self.world_size, 1)
        return self.per_device_train_batch_size * n_gpu * self.gradient_accumulation_steps


# 默认配置实例
default_config = TrainConfig()
