"""
数据集划分脚本：读取清洗后的数据，分层采样划分 8:1:1。

划分策略：
- 比例: 8:1:1（训练:验证:测试）
- 训练集目标: ~30K 条
- 按类别均匀采样：5 个类别各采 1/5
- 每类内 human:AI = 1:1
- 分层划分：按类别逐层划分，保证各 split 中类别分布和 human/AI 比例一致
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子
RANDOM_SEED = 42

# 训练集目标大小
TARGET_TRAIN_SIZE = 30000


# 类别文件名（不是 train/val/test 这类分割产物）
CATEGORY_NAMES = {"composition", "film_review", "news", "paper", "question_answer"}


def load_all_data(input_dir: Path) -> dict[str, list[dict]]:
    """加载所有清洗后的 JSONL 文件，按类别分组（排除 train/val/test.jsonl）。"""
    all_data: dict[str, list[dict]] = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        category = jsonl_file.stem
        if category not in CATEGORY_NAMES:
            continue
        records = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        all_data[category] = records
        human_count = sum(1 for r in records if r["label"] == 1)
        ai_count = sum(1 for r in records if r["label"] == 0)
        logger.info(f"加载 {category}: {len(records)} 条 (Human: {human_count}, AI: {ai_count})")
    return all_data


def stratified_split_by_category(
    records: list[dict],
    train_size: int,
    random_state: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    对单个类别的数据做分层划分。

    先按 label 分组，再从各组中分别采样 train/val/test，
    保证各 split 中 Human:AI 比例一致。
    """
    random.seed(random_state)

    human_records = [r for r in records if r["label"] == 1]
    ai_records = [r for r in records if r["label"] == 0]

    logger.info(f"  当前类别 Human: {len(human_records)}, AI: {len(ai_records)}")

    # 确定每组的 train/val/test 数量，保持比例
    # val 和 test 也强制 1:1：以 human 数量为准，AI 取等量
    human_train = int(len(human_records) * TRAIN_RATIO)
    human_val = int(len(human_records) * VAL_RATIO)
    human_test = len(human_records) - human_train - human_val

    ai_val = min(human_val, int(len(ai_records) * VAL_RATIO))
    ai_test = min(human_test, len(ai_records) - ai_val - int(len(ai_records) * TRAIN_RATIO))
    ai_train = len(ai_records) - ai_val - ai_test

    random.shuffle(human_records)
    random.shuffle(ai_records)

    train = human_records[:human_train] + ai_records[:ai_train]
    val = human_records[human_train:human_train + human_val] + ai_records[ai_train:ai_train + ai_val]
    test = human_records[human_train + human_val:] + ai_records[ai_train + ai_val:]

    logger.info(f"  划分结果: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


def sample_balanced(
    records: list[dict],
    target_human_per_cat: int,
    random_state: int,
) -> list[dict]:
    """
    从单个类别中均匀采样，保持 Human:AI = 1:1。

    当某类样本不足时，以较少一方数量为准，保证 1:1 平衡。
    """
    random.seed(random_state)
    human_records = [r for r in records if r["label"] == 1]
    ai_records = [r for r in records if r["label"] == 0]

    # 以 Human 和 AI 中较少的一方为准，确保 1:1
    n_human = min(target_human_per_cat, len(human_records))
    n_ai = min(target_human_per_cat, len(ai_records))
    n_per_label = min(n_human, n_ai)

    random.shuffle(human_records)
    random.shuffle(ai_records)

    sampled = human_records[:n_per_label] + ai_records[:n_per_label]
    random.shuffle(sampled)
    return sampled


def split_and_save() -> dict:
    """主划分流程。"""
    if not INPUT_DIR.exists():
        logger.error(f"数据目录不存在: {INPUT_DIR}")
        logger.error("请先运行 python src/data/clean.py 进行数据清洗")
        return {}

    all_data = load_all_data(INPUT_DIR)

    if not all_data:
        logger.error("未找到清洗后的数据")
        return {}

    # 计算每个类别训练、验证、测试的总数
    num_categories = len(all_data)
    train_per_cat = TARGET_TRAIN_SIZE // num_categories  # ~6000 条/类
    human_per_cat = train_per_cat // 2                    # ~3000 human + ~3000 AI
    logger.info(f"目标训练集: {TARGET_TRAIN_SIZE} 条, {num_categories} 个类别, "
                 f"每类约 {train_per_cat} 条 (human:ai = 1:1)")

    train_all: list[dict] = []
    val_all: list[dict] = []
    test_all: list[dict] = []

    stats: dict = {}

    for category, records in sorted(all_data.items()):
        logger.info(f"\n处理类别: {category} (共 {len(records)} 条)")

        # 先把整类数据按 8:1:1 分层划分
        train_cat, val_cat, test_cat = stratified_split_by_category(
            records, train_per_cat, RANDOM_SEED,
        )

        # 训练集做均匀采样（保持 1:1 平衡）
        train_sampled = sample_balanced(train_cat, human_per_cat, RANDOM_SEED + 1)
        logger.info(f"  训练集采样: {len(train_cat)} → {len(train_sampled)}")

        train_all.extend(train_sampled)
        val_all.extend(val_cat)
        test_all.extend(test_cat)

        stats[category] = {
            "train": len(train_sampled),
            "val": len(val_cat),
            "test": len(test_cat),
        }

    # 打乱全局顺序
    random.seed(RANDOM_SEED)
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    # 保存为 JSONL
    splits = {
        "train.jsonl": train_all,
        "val.jsonl": val_all,
        "test.jsonl": test_all,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, data in splits.items():
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        human_c = sum(1 for r in data if r["label"] == 1)
        ai_c = sum(1 for r in data if r["label"] == 0)
        logger.info(f"\n保存 {filename}: {len(data)} 条 (Human: {human_c}, AI: {ai_c})")

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("数据集划分完成！汇总统计：")
    logger.info("=" * 60)
    for cat, s in stats.items():
        logger.info(f"  {cat}: train={s['train']}, val={s['val']}, test={s['test']}")
    logger.info(f"  总计: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")

    return stats


if __name__ == "__main__":
    split_and_save()
