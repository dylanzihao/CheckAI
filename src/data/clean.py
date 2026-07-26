"""
数据清洗脚本：读取 raw/C-ReD 所有 CSV，清洗后按类别保存为 JSONL。

清洗步骤：
1. 去除空文本或极短文本（长度 < 10 字符）
2. 去除重复文本（基于 text 去重）
3. 统一列名和格式，human 数据补齐 prompt 列为空
4. 统一 label 字段（int 类型）
5. 去除 HTML 标签、多余空白字符
6. 确保所有 CSV 列结构一致
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 原始数据目录
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "C-ReD" / "benchmark data"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 统一输出列
UNIFIED_COLUMNS = ["id", "text", "label", "category", "type", "length",
                   "attribution", "title", "prompt"]

# HTML 标签正则
HTML_TAG_RE = re.compile(r"<[^>]+>")
# 多余空白字符正则
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """清洗单条文本：去除 HTML 标签、合并多余空白、去除首尾空格。"""
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_csv(filepath: Path) -> pd.DataFrame:
    """加载单个 CSV 文件，返回标准化的 DataFrame。"""
    df = pd.read_csv(filepath)
    logger.debug(f"  读取 {filepath.name}: {len(df)} 条记录, 列: {list(df.columns)}")
    return df


def normalize_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    标准化 DataFrame 为统一格式。

    human 数据（有 year 列，无 prompt 列） → 补齐 prompt 为空
    AI 数据（有 prompt 和 original_id 列）→ 直接映射
    """
    # 统一列映射
    result = pd.DataFrame()

    # id
    result["id"] = df["id"] if "id" in df.columns else range(len(df))

    # text: 清洗文本
    result["text"] = df["text"].apply(clean_text) if "text" in df.columns else ""

    # label: 统一为 int 类型 (1 = human, 0 = AI)
    if "label" in df.columns:
        result["label"] = df["label"].astype(int)
    else:
        result["label"] = -1

    # category: 使用目录名
    result["category"] = category

    # type
    result["type"] = df["type"] if "type" in df.columns else None

    # length
    result["length"] = df["length"] if "length" in df.columns else None

    # attribution
    result["attribution"] = df["attribution"] if "attribution" in df.columns else None

    # title: 来自 composition_title 列（两种数据都有）
    if "composition_title" in df.columns:
        result["title"] = df["composition_title"]
    else:
        result["title"] = None

    # prompt: AI 数据有，human 数据没有 → 补齐为空字符串
    if "prompt" in df.columns:
        result["prompt"] = df["prompt"].fillna("")
    else:
        result["prompt"] = ""

    return result


def clean_and_save() -> dict[str, int]:
    """
    主清洗流程：遍历所有类别和 CSV，清洗后按类别保存 JSONL。

    Returns:
        每个类别的最终记录数统计
    """
    if not RAW_DIR.exists():
        logger.error(f"原始数据目录不存在: {RAW_DIR}")
        return {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}  # category → stats dict

    for category_dir in sorted(RAW_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name.replace(" ", "_")  # e.g. "film_review", "question_answer"
        logger.info(f"处理类别: {category} ({category_dir})")

        all_records: list[dict] = []
        category_stats = {"total_raw": 0, "after_empty_removal": 0, "after_dedup": 0}

        csv_files = sorted(category_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"  类别 {category} 中没有 CSV 文件")
            continue

        for csv_file in csv_files:
            df_raw = load_csv(csv_file)
            category_stats["total_raw"] += len(df_raw)
            df_norm = normalize_df(df_raw, category)
            all_records.extend(df_norm.to_dict(orient="records"))

        logger.info(f"  原始记录数: {category_stats['total_raw']}")

        # 1. 去除空文本或极短文本（长度 < 10 字符）
        before = len(all_records)
        all_records = [r for r in all_records
                       if isinstance(r["text"], str) and len(r["text"]) >= 10]
        category_stats["after_empty_removal"] = len(all_records)
        logger.info(f"  去除空/短文本: {before} → {len(all_records)} "
                     f"(移除 {before - len(all_records)} 条)")

        # 2. 去除重复文本（基于 text 去重）
        before = len(all_records)
        seen_texts = set()
        deduped: list[dict] = []
        for r in all_records:
            if r["text"] not in seen_texts:
                seen_texts.add(r["text"])
                deduped.append(r)
        all_records = deduped
        category_stats["after_dedup"] = len(all_records)
        logger.info(f"  去除重复: {before} → {len(all_records)} "
                     f"(移除 {before - len(all_records)} 条)")

        # 3. 保存为 JSONL
        output_path = OUTPUT_DIR / f"{category}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for record in all_records:
                # 确保只输出统一列
                clean_record = {col: record.get(col) for col in UNIFIED_COLUMNS}
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

        human_count = sum(1 for r in all_records if r["label"] == 1)
        ai_count = sum(1 for r in all_records if r["label"] == 0)
        logger.info(f"  保存到: {output_path} (共 {len(all_records)} 条, "
                     f"Human: {human_count}, AI: {ai_count})")

        stats[category] = category_stats

    # 汇总统计
    logger.info("\n" + "=" * 60)
    logger.info("数据清洗完成！汇总统计：")
    logger.info("=" * 60)
    total_cleaned = 0
    for cat, s in stats.items():
        logger.info(f"  {cat}: 原始 {s['total_raw']} → 清洗后 {s['after_dedup']}")
        total_cleaned += s["after_dedup"]
    logger.info(f"  总计清洗后: {total_cleaned} 条")
    logger.info(f"  输出目录: {OUTPUT_DIR}")

    return {cat: s["after_dedup"] for cat, s in stats.items()}


if __name__ == "__main__":
    clean_and_save()
