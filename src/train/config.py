"""
训练配置：路径解析 + HuggingFace TrainingArguments。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from transformers import TrainingArguments


@dataclass
class TrainConfig:
    """训练路径与模型参数（兼容本地和 Kaggle 环境）。"""

    # --- 模型路径（优先级：本地 > Kaggle 缓存 > Kaggle 输入 > HuggingFace ID）---
    pretrained_model_name: str = "hfl/chinese-roberta-wwm-ext"
    local_model_dir: str = "D:\\Dylan\\Model\\chinese-roberta-wwm-ext"
    kaggle_model_dir: str = "/kaggle/input/models/dylanzihao/chinese-bert-wwm-ext/transformers/default/1"

    # 数据目录
    kaggle_data_dir: str = "/kaggle/input/datasets/dylanzihao/c-red-processed"
    data_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed",
    ))

    # 输出目录
    output_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent.parent / "models" / "base",
    ))

    # --- 模型参数 ---
    max_seq_length: int = 512
    num_labels: int = 2

    # --- 训练参数（传给 TrainingArguments）---
    seed: int = 42
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    fp16: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True

    def resolve_model_name(self) -> str:
        """解析模型路径：本地 > Kaggle 缓存 > Kaggle 输入 > HuggingFace ID。"""
        if Path(self.local_model_dir).exists():
            return self.local_model_dir
        kaggle_cached = "/kaggle/working/models/pretrained"
        if Path(kaggle_cached).exists():
            return kaggle_cached
        if Path(self.kaggle_model_dir).exists():
            return self.kaggle_model_dir
        return self.pretrained_model_name

    def resolve_data_dir(self) -> str:
        """解析数据目录：Kaggle 输入 > 项目本地路径。"""
        if Path(self.kaggle_data_dir).exists():
            return self.kaggle_data_dir
        return self.data_dir

    def make_training_args(self) -> TrainingArguments:
        """从 TrainConfig 创建 TrainingArguments。"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=self.num_train_epochs,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            seed=self.seed,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            load_best_model_at_end=self.load_best_model_at_end,
            report_to=[],
            remove_unused_columns=False,
        )


default_config = TrainConfig()
