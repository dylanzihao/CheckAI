"""
数据增强脚本：对训练集进行 nlpcda 数据增强（多进程并行）。

增强方法：
- Similarword: 同/近义词替换 (需 --full 启用，速度较慢)
- Randomword: 随机词替换
- RandomDeleteChar: 随机删除字符
- CharPositionExchange: 交换字符位置
- EquivalentChar: 等价字替换
- Homophone: 同音字替换

增强倍率：每个原始样本生成 2 个增强变体。
注意事项：
- 增强后的训练集需保持 1:1 平衡
- 增强样本的 label 与原始样本一致
- 验证集和测试集不做增强

用法:
    python src/data/augment.py                        # 全部训练集增强（默认 8 进程）
    python src/data/augment.py --workers 16            # 16 进程并行
    python src/data/augment.py --max-samples 500       # 仅增强 500 条（快速测试）
    python src/data/augment.py --full                  # 启用 Similarword
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "augmented" / "train.jsonl"

RANDOM_SEED = 42
NUM_AUG_PER_SAMPLE = 2
TARGET_TRAIN_SIZE = 30000  # 增强后训练集目标大小
CHUNK_SIZE_PER_WORKER = 30  # 每个 worker 一次处理的记录数（越小越能负载均衡）


def load_train_data(filepath: Path) -> list[dict]:
    """加载训练集 JSONL。"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_augmented_data(records: list[dict], filepath: Path) -> None:
    """保存增强后的数据为 JSONL。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---- Worker 进程中的全局变量 ----
_worker_augmenters = None


def _init_worker(full: bool, seed: int):
    """Worker 进程初始化：设置随机种子 + 初始化增强器。"""
    import warnings
    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)

    global _worker_augmenters
    _worker_augmenters = {}

    if full:
        try:
            from nlpcda import Similarword
            _worker_augmenters["similarword"] = Similarword(create_num=3)
        except Exception:
            pass

    try:
        from nlpcda import Randomword
        _worker_augmenters["randomword"] = Randomword(create_num=3)
    except Exception:
        pass

    try:
        from nlpcda import RandomDeleteChar
        _worker_augmenters["delete_char"] = RandomDeleteChar(create_num=3)
    except Exception:
        pass

    try:
        from nlpcda import CharPositionExchange
        _worker_augmenters["char_swap"] = CharPositionExchange(create_num=3)
    except Exception:
        pass

    try:
        from nlpcda import EquivalentChar
        _worker_augmenters["equiv_char"] = EquivalentChar(create_num=3)
    except Exception:
        pass

    try:
        from nlpcda import Homophone
        _worker_augmenters["homophone"] = Homophone(create_num=3)
    except Exception:
        pass


def _augment_chunk(records: list[dict]) -> list[dict]:
    """
    Worker 函数：对一个 chunk 进行增强，返回 [原始, 增强1, 增强2, ...]。
    每个原始样本生成 ~NUM_AUG_PER_SAMPLE 个变体。
    """
    global _worker_augmenters
    augmenters = _worker_augmenters
    if not augmenters:
        return records  # 无增强器可用，返回原始数据

    results = []
    for record in records:
        results.append(record)  # 保留原始

        text = record.get("text", "")
        if len(text) < 10:
            continue

        # 收集所有增强器的变体
        all_augmented: list[str] = []
        for aug in augmenters.values():
            try:
                aug_results = aug.replace(text)
                if isinstance(aug_results, list):
                    all_augmented.extend(
                        [r for r in aug_results if r and r != text]
                    )
                elif isinstance(aug_results, str) and aug_results and aug_results != text:
                    all_augmented.append(aug_results)
            except Exception:
                pass

        # 随机选择 NUM_AUG_PER_SAMPLE 个
        if len(all_augmented) > NUM_AUG_PER_SAMPLE:
            random.shuffle(all_augmented)
            all_augmented = all_augmented[:NUM_AUG_PER_SAMPLE]

        # 去重并生成增强记录
        for aug_text in dict.fromkeys(all_augmented):
            if len(aug_text) < 10:
                continue
            new_record = record.copy()
            new_record["text"] = aug_text
            results.append(new_record)

    return results


def _split_into_chunks(records: list[dict], num_workers: int) -> list[list[dict]]:
    """将记录列表拆分为适合多进程处理的 chunks。"""
    chunk_size = max(CHUNK_SIZE_PER_WORKER, len(records) // (num_workers * 2))
    chunks = []
    for i in range(0, len(records), chunk_size):
        chunks.append(records[i : i + chunk_size])
    return chunks


def augment(
    max_samples: int | None = None,
    full: bool = False,
    workers: int | None = None,
) -> None:
    """主增强流程（多进程并行）。"""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("开始数据增强（多进程并行）")
    logger.info("=" * 60)

    if not TRAIN_FILE.exists():
        logger.error(f"训练集文件不存在: {TRAIN_FILE}")
        logger.error("请先运行 python src/data/split.py 进行数据划分")
        return

    # 加载训练集
    train_data = load_train_data(TRAIN_FILE)
    logger.info(f"加载训练集: {len(train_data)} 条")

    # 分离 Human 和 AI，随机采样
    human_records = [r for r in train_data if r["label"] == 1]
    ai_records = [r for r in train_data if r["label"] == 0]

    if max_samples is not None:
        full_total = len(human_records) + len(ai_records)  # 采样前的总数
        half = max_samples // 2
        random.seed(RANDOM_SEED)
        random.shuffle(human_records)
        random.shuffle(ai_records)
        human_records = human_records[:half]
        ai_records = ai_records[:half]
        # 等比例缩放目标
        scale = max_samples / full_total
        target_size = int(TARGET_TRAIN_SIZE * scale)
    else:
        target_size = TARGET_TRAIN_SIZE

    human_count = len(human_records)
    ai_count = len(ai_records)
    current_total = human_count + ai_count
    logger.info(f"  原始训练集: {current_total} 条 (Human: {human_count}, AI: {ai_count})")

    # 计算需要增强多少样本才能达到目标大小
    gap = max(0, target_size - current_total)
    # 每个样本生成 NUM_AUG_PER_SAMPLE 个变体 → 输出 = 1(原) + NUM_AUG_PER_SAMPLE
    multiplier = 1 + NUM_AUG_PER_SAMPLE  # 3x
    aug_needed_per_label = gap // (2 * NUM_AUG_PER_SAMPLE)  # 每类需增强的原始样本数

    if aug_needed_per_label <= 0:
        logger.info(f"  无需增强，当前 {current_total} 条已达到目标 {target_size}")
        all_augmented_data = list(train_data)
        random.shuffle(all_augmented_data)
        save_augmented_data(all_augmented_data, OUTPUT_FILE)
        return

    # 确保不超过可用数量
    aug_needed_per_label = min(aug_needed_per_label, human_count, ai_count)
    actual_total = current_total + aug_needed_per_label * 2 * NUM_AUG_PER_SAMPLE
    logger.info(f"  目标: ~{target_size} 条, 需增强 {aug_needed_per_label} 条/类")
    logger.info(f"  增强后预计: {actual_total} 条")

    # 确定 worker 数量
    cpu_count = os.cpu_count() or 4
    if workers is None:
        workers = min(cpu_count - 1, 12)  # 留一个核给系统，最多 12
    workers = max(1, workers)
    logger.info(f"  并行进程数: {workers} (CPU: {cpu_count})")

    # 分别处理 Human 和 AI，保持平衡
    all_augmented_data: list[dict] = []

    for label, records, label_name in [
        (1, human_records, "Human"),
        (0, ai_records, "AI"),
    ]:
        # 随机选取需要增强的样本，其余直接保留
        random.seed(RANDOM_SEED + label)
        shuffled = list(records)
        random.shuffle(shuffled)
        to_augment = shuffled[:aug_needed_per_label]
        keep_as_is = shuffled[aug_needed_per_label:]

        logger.info(f"\n{'─' * 40}")
        logger.info(f"增强 {label_name} 数据: 增强 {len(to_augment)} 条 + "
                     f"保留 {len(keep_as_is)} 条")

        # 未增强的原始数据直接添加
        all_augmented_data.extend(keep_as_is)

        if len(to_augment) == 0:
            continue

        # 拆分需要增强的记录为 chunks
        chunks = _split_into_chunks(to_augment, workers)
        worker_seeds = [RANDOM_SEED + label * 10000 + i * 1000 for i in range(workers)]
        logger.info(f"  拆分 {len(chunks)} 个 chunks, 提交到 {workers} 个 worker")

        aug_count = 0
        chunks_done = 0

        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(full, worker_seeds[0]),
        ) as pool:
            for chunk_results in pool.imap_unordered(_augment_chunk, chunks):
                chunk_orig_count = len(chunk_results)  # 这个chunk输出的总条数（含原始+增强）
                # 计算这个 chunk 中的原始记录数：排除增强变体（变体数量不超过原始记录数*NUM_AUG_PER_SAMPLE）
                # 直接用 all_augmented_data 中的唯一原始 id 来推算
                all_augmented_data.extend(chunk_results)
                chunks_done += 1

                # 根据总条数推算进度
                # 每个 chunk 原始记录 ~CHUNK_SIZE_PER_WORKER，增强后 3x 输出
                est_orig = len(all_augmented_data) // 3
                aug_count = len(all_augmented_data) - est_orig

                if chunks_done % 5 == 0 or chunks_done == len(chunks):
                    pct = chunks_done / len(chunks) * 100
                    logger.info(
                        f"  [{label_name}] {chunks_done}/{len(chunks)} chunks "
                        f"({pct:.0f}%), 当前累计 {len(all_augmented_data)} 条"
                    )

        logger.info(f"  {label_name} 完成: {len(to_augment)} 增强 + {len(keep_as_is)} 保留, "
                     f"{chunks_done} chunks 处理完毕")

    # 最终统计
    final_human = sum(1 for r in all_augmented_data if r["label"] == 1)
    final_ai = sum(1 for r in all_augmented_data if r["label"] == 0)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("增强完成统计：")
    logger.info(f"  原始: {human_count + ai_count} 条 (Human: {human_count}, AI: {ai_count})")
    logger.info(f"  增强后: {len(all_augmented_data)} 条 (Human: {final_human}, AI: {final_ai})")
    logger.info(f"  耗时: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"  速度: {len(all_augmented_data) / elapsed:.0f} 条/秒")

    # 打乱并保存
    random.seed(RANDOM_SEED)
    random.shuffle(all_augmented_data)

    save_augmented_data(all_augmented_data, OUTPUT_FILE)
    logger.info(f"  保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    # Windows 下 multiprocessing 需要 freeze_support
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="数据增强（多进程并行）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制用于增强的样本数量（快速测试用）")
    parser.add_argument("--full", action="store_true",
                        help="启用 Similarword 增强器（速度较慢但质量更高）")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数（默认 CPU 核心数-1，最大 12）")
    args = parser.parse_args()

    augment(max_samples=args.max_samples, full=args.full, workers=args.workers)
