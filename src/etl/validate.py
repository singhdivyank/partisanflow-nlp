"""Per-year data quality checks"""

from pyspark.sql import DataFrame, functions as F

from src.utils.constants import (
    COL_CLEANED,
    COL_RAW_LABEL, 
    COL_ISSUE,
    COL_SERIES,
    COL_PARAGRAPH_ID,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

def compute_basic_stats(df: DataFrame, year: int) -> dict[str, object]:
    """Compute and log basic stats for a year's processed DataFrame to be logged to MLflow"""

    stats = {}

    agg = df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct(COL_SERIES).alias("distinct_series"),
        F.countDistinct(COL_SERIES, COL_ISSUE).alias("distinct_issues"),
    ).first()

    stats["year"] = year
    stats["row_count"] = agg["row_count"]
    stats["distinct_series"] = agg["distinct_series"]
    stats["distinct_issues"] = agg["distinct_issues"]

    log.info(
        "[validate | year=%d] rows=%d, series=%d, issues=%d",
        year, 
        stats["row_count"], 
        stats["distinct_series"], 
        stats["distinct_issues"]
    )

    for col in [COL_SERIES, COL_ISSUE, COL_PARAGRAPH_ID]:
        n_null = df.filter(F.col(col).isNull()).count()
        stats[f"null_{col}"] = n_null
        if n_null:
            log.warning(
                "[validate | year=%d] '%s' has %d null(s).", 
                year, col, n_null
            )
    
    len_stats = get_para_len_stats(df)
    stats.update(len_stats)

    log.info(
        "[validate | year=%d] Paragraph lengths — min=%d, max=%d, avg=%.1f",
        year,
        stats["min_para_len"],
        stats["max_para_len"],
        stats["avg_para_len"],
    )

    if year == 1869:
        label_distribution = get_label_distribution(df)
        stats["label_distribution"] = label_distribution
        log.info("[validate | year=1869] Label distribution: %s", label_distribution)
    
    return stats

def get_para_len_stats(df: DataFrame) -> dict:
    """Stats for the text/para lengths"""

    min_len, max_len, avg_len = 0, 0, 0

    if COL_CLEANED in df.columns:
        agg_len = df.agg(
            F.min(F.length(COL_CLEANED)).alias("min_length"),
            F.max(F.length(COL_CLEANED)).alias("max_length"),
            F.avg(F.length(COL_CLEANED)).alias("avg_len")
        ).first()
        min_len = agg_len["min_length"]
        max_len = agg_len["max_length"]
        avg_len = round(agg_len["avg_len"], 1)
    
    if max_len > 50_000:
        log.warning("Extreme paragraph length detected: %d chars.", max_len)

    return {
        "min_para_len": min_len,
        "max_para_len": max_len,
        "avg_para_len": avg_len
    }

def get_label_distribution(df: DataFrame) -> dict:
    """Obtain label distribution for the year 1869"""

    dist = (
        df.groupBy(COL_RAW_LABEL)
        .count()
        .orderBy(COL_RAW_LABEL)
        .collect()
    )
    label_counts = {row[COL_RAW_LABEL]: row["count"] for row in dist}

    # label imbalance
    counts = [v for v in label_counts.values() if v]
    if len(counts) >= 2:
        ratio = max(counts) / min(counts)
        if ratio > 4.0:
            log.warning(
                "[validate | year=1869] High class imbalance detected (ratio=%.2f)",
                ratio
            )

    return label_counts

def check_train_labels_present(df: DataFrame, year: int) -> bool:
    """Verify that both Democratic and Republican labels appear in training DataFrame"""

    if COL_RAW_LABEL not in df.columns:
        log.error(
            "[validate | year=%d] Label column '%s' not found.", 
            year, 
            COL_RAW_LABEL
        )
        return False
    
    present = {
        row[COL_RAW_LABEL]
        for row in df.select(COL_RAW_LABEL).distinct().collect()
    }
    missing = [lbl for lbl in [0, 1] if lbl not in present]

    if missing:
        log.error(
            "[validate | year=%d] Training labels %s not found in data.", 
            year, 
            missing
        )
        return False
    
    log.info("[validate | year=%d] All training labels present.", year)
    return True

def validate_year(
    df: DataFrame, 
    year: int, 
    is_training: bool = False
) -> bool:
    """Orchestrate quality checks for a year"""

    ok = True
    stats = compute_basic_stats(df, year)

    if not stats["row_count"]:
        log.error("[validate | year=%d] DataFrame is empty", year)
        ok = False
    
    if is_training:
        if not check_train_labels_present(df, year):
            ok = False

    if not ok:
        raise ValueError(f"Data validation failed for year={year}. See logs for details.")

    return ok
