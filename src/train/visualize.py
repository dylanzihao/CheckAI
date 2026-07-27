"""
训练成果可视化：使用 matplotlib 生成训练指标图表，静默保存为 PNG。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

BLUE = "#2c7bb6"
RED = "#d7191c"
GREEN = "#1a9641"
PURPLE = "#762a83"


def _softmax(logits: np.ndarray) -> np.ndarray:
    """稳定的 softmax，axis=1。"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _extract_loss_data(
    log_history: list[dict],
) -> tuple[list, list, list, list]:
    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    for entry in log_history:
        step = entry.get("step", 0)
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])
    return train_steps, train_losses, eval_steps, eval_losses


def _extract_metrics_data(
    log_history: list[dict],
) -> tuple[list, list, list, list, list]:
    steps: list[int] = []
    accuracy: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for entry in log_history:
        if "eval_f1" in entry:
            steps.append(entry.get("step", 0))
            accuracy.append(entry["eval_accuracy"])
            precision.append(entry["eval_precision"])
            recall.append(entry["eval_recall"])
            f1.append(entry["eval_f1"])
    return steps, accuracy, precision, recall, f1


def plot_loss_curves(log_history: list[dict], save_path: Path) -> None:
    train_steps, train_losses, eval_steps, eval_losses = _extract_loss_data(log_history)
    if not train_steps and not eval_steps:
        logger.warning("log_history 中无 loss 数据，跳过 loss_curves.png")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_steps:
        ax.plot(train_steps, train_losses, label="Train Loss", color=BLUE, alpha=0.8)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color=RED, marker="o", markersize=4)
        best_idx = int(np.argmin(eval_losses))
        ax.annotate(
            f"{eval_losses[best_idx]:.4f}",
            (eval_steps[best_idx], eval_losses[best_idx]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=9,
            color=RED,
        )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Evaluation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {save_path}")


def plot_metrics_curves(log_history: list[dict], save_path: Path) -> None:
    steps, accuracy, precision, recall, f1 = _extract_metrics_data(log_history)
    if not steps:
        logger.warning("log_history 中无 eval 指标数据，跳过 metrics_curves.png")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, accuracy, label="Accuracy", color=BLUE, marker="s", markersize=3)
    ax.plot(steps, precision, label="Precision", color=RED, marker="^", markersize=3)
    ax.plot(steps, recall, label="Recall", color=GREEN, marker="v", markersize=3)
    ax.plot(steps, f1, label="F1", color=PURPLE, marker="o", markersize=3)

    best_idx = int(np.argmax(f1))
    ax.annotate(
        f"Best F1: {f1[best_idx]:.4f}",
        (steps[best_idx], f1[best_idx]),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        fontsize=9,
        color=PURPLE,
    )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            text = f"{cm[i, j]}\n({pct:.1f}%)"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["AI-Generated", "Human"])
    ax.set_yticklabels(["AI-Generated", "Human"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {save_path}")


def plot_dashboard(
    log_history: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: Path,
) -> None:
    train_steps, train_losses, eval_steps, eval_losses = _extract_loss_data(log_history)
    metrics_steps, accuracy, precision, recall, f1 = _extract_metrics_data(log_history)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = float(auc(fpr, tpr))

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 左上：Loss 曲线
    ax = axes[0, 0]
    if train_steps:
        ax.plot(train_steps, train_losses, label="Train Loss", color=BLUE, alpha=0.8)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color=RED, marker="o", markersize=4)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Evaluation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右上：指标曲线
    ax = axes[0, 1]
    if metrics_steps:
        ax.plot(metrics_steps, accuracy, label="Accuracy", color=BLUE, marker="s", markersize=3)
        ax.plot(metrics_steps, precision, label="Precision", color=RED, marker="^", markersize=3)
        ax.plot(metrics_steps, recall, label="Recall", color=GREEN, marker="v", markersize=3)
        ax.plot(metrics_steps, f1, label="F1", color=PURPLE, marker="o", markersize=3)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 左下：混淆矩阵
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            text = f"{cm[i, j]}\n({pct:.1f}%)"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["AI-Generated", "Human"])
    ax.set_yticklabels(["AI-Generated", "Human"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 右下：ROC 曲线
    ax = axes[1, 1]
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  已保存: {save_path}")


def visualize_all(
    log_history: list[dict],
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: str | Path,
) -> None:
    """训练结束后调用，生成全部 5 张 PNG 图表。"""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.argmax(y_probs, axis=1)
    prob_human = y_probs[:, 1]

    logger.info("生成训练可视化图表...")
    plot_loss_curves(log_history, plots_dir / "loss_curves.png")
    plot_metrics_curves(log_history, plots_dir / "metrics_curves.png")
    plot_confusion_matrix(y_true, y_pred, plots_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, prob_human, plots_dir / "roc_curve.png")
    plot_dashboard(log_history, y_true, y_pred, prob_human, plots_dir / "dashboard.png")
    logger.info("可视化图表生成完毕。")
