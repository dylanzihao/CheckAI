"""
训练主逻辑：支持单机单卡 / 单机多卡（DDP）+ FP16 混合精度训练。

执行方式：
    # 单卡训练
    python src/train/trainer.py

    # 多卡训练（torchrun / torch.distributed.launch）
    torchrun --nproc_per_node=2 src/train/trainer.py
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path

# 确保项目根目录在 Python path 中（兼容 torchrun 子进程）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import RandomSampler as _RandomSampler
from torch.utils.data import SequentialSampler as _SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.train.config import TrainConfig, default_config
from src.train.dataset import AITextDataset
from src.train.model import RobertaForAITextDetection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """固定随机种子，确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(config: TrainConfig) -> None:
    """初始化分布式训练环境。"""
    if config.is_distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        logger.info(
            f"分布式训练初始化完成: rank={config.local_rank}, "
            f"world_size={config.world_size}",
        )


def cleanup_distributed() -> None:
    """清理分布式训练环境。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """计算分类指标。"""
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    config: TrainConfig,
) -> dict[str, float]:
    """在验证集上评估模型。"""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(config.device)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs["logits"]
            total_loss += loss_fn(logits, labels).item()

            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = avg_loss

    return metrics


def train(config: TrainConfig | None = None) -> None:
    """完整训练流程。"""
    if config is None:
        config = default_config

    # ---- 分布式初始化 ----
    setup_distributed(config)
    set_seed(config.seed)

    if config.is_main_process:
        logger.info("=" * 60)
        logger.info("AI 文本检测模型训练")
        logger.info("=" * 60)
        logger.info(f"模型: {config.pretrained_model_name}")
        logger.info(f"本地路径: {config.local_model_dir}")
        logger.info(f"最大序列长度: {config.max_seq_length}")
        logger.info(f"Per-GPU batch size: {config.per_device_train_batch_size}")
        logger.info(f"有效 batch size: {config.effective_batch_size}")
        logger.info(f"学习率: {config.learning_rate}")
        logger.info(f"训练轮数: {config.num_epochs}")
        logger.info(f"FP16: {config.fp16}")
        logger.info(f"分布式: {config.is_distributed} (world_size={config.world_size})")
        logger.info(f"梯度累积步数: {config.gradient_accumulation_steps}")
        logger.info(f"Warmup 比例: {config.warmup_ratio}")
        logger.info("=" * 60)

    # ---- 加载 Tokenizer ----
    model_name = config.resolve_model_name()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    _pin_memory = torch.cuda.is_available()

    # ---- 加载数据 ----
    data_dir = config.resolve_data_dir()
    train_file = Path(data_dir) / "train.jsonl"
    val_file = Path(data_dir) / "val.jsonl"
    test_file = Path(data_dir) / "test.jsonl"

    train_dataset = AITextDataset(train_file, tokenizer, config.max_seq_length)
    val_dataset = AITextDataset(val_file, tokenizer, config.max_seq_length)

    # 分布式采样器
    if config.is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = _RandomSampler(train_dataset)
        val_sampler = _SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=_pin_memory,
    )

    # ---- 加载模型 ----
    logger.info("初始化模型...")
    model = RobertaForAITextDetection.from_pretrained_with_head(
        model_name_or_path=model_name,
        num_labels=config.num_labels,
        dropout_rate=config.dropout_rate,
    )
    model.to(config.device)

    # 分布式包装
    if config.is_distributed:
        model = DDP(model, device_ids=[config.local_rank], find_unused_parameters=True)
        base_model = model.module
    else:
        base_model = model

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    # ---- 损失函数（DDP 要求在模型外部计算 loss） ----
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- 学习率调度器 ----
    total_steps = (
        math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"总训练步数: {total_steps}, warmup 步数: {warmup_steps}")

    # ---- FP16 混合精度 ----
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    # ---- 训练循环 ----
    global_step = 0
    best_metric = 0.0 if config.greater_is_better else float("inf")

    for epoch in range(config.num_epochs):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        optimizer.zero_grad()

        if config.is_main_process:
            logger.info(f"\n{'=' * 40}\nEpoch {epoch + 1}/{config.num_epochs}\n{'=' * 40}")

        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(config.device)
            batch = {k: v.to(config.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = model(**batch)
                loss = loss_fn(outputs["logits"], labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), config.max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # 日志
                if config.is_main_process and global_step % config.logging_steps == 0:
                    steps_since_log = config.logging_steps * config.gradient_accumulation_steps
                    avg_loss = epoch_loss / steps_since_log
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}",
                    )
                    epoch_loss = 0.0

                # 评估
                if config.is_main_process and global_step % config.eval_steps == 0:
                    logger.info(f"  --- 评估 @ Step {global_step} ---")
                    val_metrics = evaluate(model, val_loader, loss_fn, config)
                    logger.info(
                        f"  Val | Loss: {val_metrics['loss']:.4f} | "
                        f"Acc: {val_metrics['accuracy']:.4f} | "
                        f"P: {val_metrics['precision']:.4f} | "
                        f"R: {val_metrics['recall']:.4f} | "
                        f"F1: {val_metrics['f1']:.4f}",
                    )

                    current_metric = val_metrics[config.metric_for_best_model]
                    is_best = (
                        current_metric > best_metric
                        if config.greater_is_better
                        else current_metric < best_metric
                    )

                    if is_best:
                        best_metric = current_metric
                        save_model(base_model, tokenizer, config.output_dir, best=True)
                        logger.info(f"  >>> 保存最佳模型 (F1={best_metric:.4f})")

                    model.train()

        # Handle remaining accumulated gradients at end of epoch
        if (step + 1) % config.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                base_model.parameters(), config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Epoch 结束评估
        if config.is_main_process:
            logger.info(f"\n--- Epoch {epoch + 1} 结束，完整评估 ---")
            val_metrics = evaluate(model, val_loader, loss_fn, config)
            logger.info(
                f"  Val | Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"P: {val_metrics['precision']:.4f} | "
                f"R: {val_metrics['recall']:.4f} | "
                f"F1: {val_metrics['f1']:.4f}",
            )

            current_metric = val_metrics[config.metric_for_best_model]
            is_best = (
                current_metric > best_metric
                if config.greater_is_better
                else current_metric < best_metric
            )

            if is_best:
                best_metric = current_metric
                save_model(base_model, tokenizer, config.output_dir, best=True)
                logger.info(f"  >>> 保存最佳模型 (F1={best_metric:.4f})")

    # ---- 最终评估 ----
    if config.is_main_process:
        logger.info(f"\n{'=' * 40}\n训练完成，最终评估\n{'=' * 40}")
        logger.info(f"最佳 {config.metric_for_best_model}: {best_metric:.4f}")

        # 加载最佳模型进行评估（使用 from_pretrained 加载完整权重，包括分类头）
        logger.info("加载最佳模型...")
        best_model = RobertaForAITextDetection.from_pretrained(
            str(Path(config.output_dir) / "best"),
        )
        best_model.to(config.device)

        # 测试集评估（如果存在）
        if test_file.exists():
            test_dataset = AITextDataset(test_file, tokenizer, config.max_seq_length)
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.per_device_eval_batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                pin_memory=_pin_memory,
            )
            test_metrics = evaluate(best_model, test_loader, loss_fn, config)
            logger.info(
                f"  Test | Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | "
                f"P: {test_metrics['precision']:.4f} | "
                f"R: {test_metrics['recall']:.4f} | "
                f"F1: {test_metrics['f1']:.4f}",
            )

        # 保存最终 epoch 模型（last checkpoint）
        last_path = Path(config.output_dir) / "last"
        last_path.mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(str(last_path))
        tokenizer.save_pretrained(str(last_path))
        logger.info(f"最终 epoch 模型已保存至: {last_path}")

    cleanup_distributed()


def save_model(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str,
    best: bool = False,
) -> None:
    """保存模型和 tokenizer。"""
    save_path = Path(output_dir)
    if best:
        save_path = save_path / "best"
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    logger.info(f"模型已保存至: {save_path}")


if __name__ == "__main__":
    train()
