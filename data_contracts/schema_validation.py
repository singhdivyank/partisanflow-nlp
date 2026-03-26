"""Data contract validation for every stage of pipeline."""

import json
from pathlib import Path

from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.ml.linalg import VectorUDT

from src.utils.constants import (
    COL_DATE,
    COL_FEATURES,
    COL_ISSUE,
    COL_RAW_LABEL,
    COL_SERIES,
    COL_TEXT,
    COL_YEAR,
    COL_PARAGRAPH_ID,
    COL_PRED_LABEL
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_EXPECTATIONS_PATH = Path(__file__).parent / "expectations.json"

# SCHEMA VALIDATORS

RAW_PARQUET_SCHEMA = T.StructType([
    T.StructField(COL_SERIES, T.StringType(),    nullable=False),
    T.StructField(COL_ISSUE,  T.StringType(),    nullable=False),
    T.StructField(COL_DATE,   T.DateType(),      nullable=True),
    T.StructField(COL_TEXT,   T.StringType(),    nullable=False),
])

METADATA_SCHEMA = T.StructType([
    T.StructField(COL_SERIES,   T.StringType(), nullable=False),
    T.StructField("contents",   T.StringType(), nullable=True),
])

PROCESSED_SCHEMA = T.StructType([
    T.StructField(COL_SERIES,       T.StringType(), nullable=False),
    T.StructField(COL_ISSUE,        T.StringType(), nullable=False),
    T.StructField(COL_YEAR,         T.IntegerType(), nullable=False),
    T.StructField(COL_PARAGRAPH_ID, T.LongType(),   nullable=False),
    T.StructField(COL_TEXT,         T.StringType(), nullable=True),
    T.StructField(COL_RAW_LABEL,        T.DoubleType(), nullable=True),
])

FEATURE_STORE_SCHEMA = T.StructType([
    T.StructField(COL_SERIES,       T.StringType(), nullable=False),
    T.StructField(COL_ISSUE,        T.StringType(), nullable=False),
    T.StructField(COL_YEAR,         T.IntegerType(), nullable=False),
    T.StructField(COL_PARAGRAPH_ID, T.LongType(),   nullable=False),
    T.StructField(COL_FEATURES,     VectorUDT(),  nullable=False),
    T.StructField(COL_RAW_LABEL,        T.DoubleType(), nullable=True),
])

# Helpers

def _load_expectations() -> dict:
    if _EXPECTATIONS_PATH.exists():
        with open(_EXPECTATIONS_PATH) as f:
            return json.load(f)
    log.warning("expectations.json not found at %s", _EXPECTATIONS_PATH)
    return {}

def _check_required_columns(df: DataFrame, required: list[str], stage: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("[%s] Missing columns: %s", stage, missing)
    return missing

def _check_nulls(df: DataFrame, not_null_cols: list[str], stage: str) -> dict:
    """Returns {col: null_count} for any column that has nulls."""
    cols = [col for col in not_null_cols if col in df.columns]
    if not cols:
        return {}
    
    agg_exprs = [
        F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias(col)
        for col in cols
    ]

    row = df.agg(*agg_exprs).first()

    results = {}
    for _col in cols:
        n = row[_col]
        if n and n > 0:
            log.warning(
                "[%s] Column '%s' has %d null(s).", 
                stage, 
                _col, 
                n
            )
            results[_col] = n
    return results

def _check_label_distribution(df: DataFrame, label_col: str, stage: str) -> None:
    if label_col not in df.columns:
        return
    dist = df.groupBy(label_col).count().orderBy(label_col)
    log.info("[%s] Label distribution:", stage)
    for row in dist.collect():
        log.info(
            "  label=%.1f  count=%d", 
            row[label_col],
            row["count"]
        )

def _check_row_count(row_count, min_rows: int, stage: str) -> bool:
    log.info("[%s] Row count: %d", stage, row_count)
    if row_count < min_rows:
        log.error(
            "[%s] Row count %d below minimum %d.", 
            stage, 
            row_count, 
            min_rows
        )
        return False
    return True

# Public validators

def validate_raw(df: DataFrame, raise_on_error: bool = False) -> bool:
    """Validate raw parquet input."""
    stage = "raw_parquet"
    exp   = _load_expectations().get(stage, {})
    ok    = True

    log.info("Validating Raw Data")

    missing = _check_required_columns(
        df,
        [COL_SERIES, COL_ISSUE, COL_TEXT],
        stage,
    )
    if missing:
        ok = False

    log.info("Missing column check completed")
    
    null_report = _check_nulls(
        df, 
        [COL_SERIES, COL_ISSUE, COL_TEXT], 
        stage
    )
    if null_report:
        ok = False

    log.info("Generated null report")

    min_rows = exp.get("min_row_count", 1)
    n = df.count()
    if not _check_row_count(n, min_rows, stage):
        ok = False

    log.info("Completed row count check")

    if not ok and raise_on_error:
        raise ValueError(f"[{stage}] Validation failed. See logs for details.")
    
    return ok


def validate_processed(df: DataFrame, raise_on_error: bool = False) -> bool:
    """Validate post-ETL processed DataFrame."""
    stage = "processed"
    exp   = _load_expectations().get(stage, {})
    ok    = True

    log.info("Validating Processed Data")
    missing = _check_required_columns(
        df,
        [COL_SERIES, COL_ISSUE, COL_YEAR, COL_PARAGRAPH_ID],
        stage,
    )
    if missing:
        ok = False

    _check_label_distribution(df, COL_RAW_LABEL, stage)

    # Check for extreme text lengths
    if COL_TEXT in df.columns:
        stats = df.select(
            F.min(F.length(COL_TEXT)).alias("min_len"),
            F.max(F.length(COL_TEXT)).alias("max_len"),
            F.avg(F.length(COL_TEXT)).alias("avg_len"),
        ).first()
        log.info(
            "[%s] Text length — min=%d, max=%d, avg=%.0f",
            stage, 
            stats.min_len or 0, 
            stats.max_len or 0, 
            stats.avg_len or 0,
        )
        max_allowed = exp.get("max_text_length", 100_000)
        if (stats.max_len or 0) > max_allowed:
            log.warning(
                "[%s] Extreme text length detected: %d", 
                stage, 
                stats.max_len
            )

    min_rows = exp.get("min_row_count", 1)
    if not _check_row_count(df, min_rows, stage):
        ok = False

    if not ok and raise_on_error:
        raise ValueError(f"[{stage}] Validation failed.")
    
    log.info("Validation successfully. No. of issues: %d", 0)
    return ok


def validate_features(df: DataFrame, raise_on_error: bool = False) -> bool:
    """Validate feature store DataFrame."""
    stage = "feature_store"
    ok    = True

    log.info("Validating Features")
    missing = _check_required_columns(
        df,
        [COL_SERIES, COL_ISSUE, COL_YEAR, COL_PARAGRAPH_ID, COL_FEATURES],
        stage,
    )
    if missing:
        ok = False

    null_report = _check_nulls(df, [COL_FEATURES], stage)
    if null_report:
        ok = False

    if not ok and raise_on_error:
        raise ValueError(f"[{stage}] Validation failed.")
    
    log.info("Validation successfully. No. of issues: %d", 0)
    return ok


def validate_predictions(df: DataFrame, raise_on_error: bool = False) -> bool:
    """Validate batch prediction output."""
    stage = "predictions"
    ok    = True

    log.info("Validating predictions")
    missing = _check_required_columns(
        df,
        [COL_SERIES, COL_ISSUE, COL_YEAR, COL_PRED_LABEL],
        stage,
    )
    if missing:
        ok = False

    # Warn if any label outside expected range
    if COL_PRED_LABEL in df.columns:
        unexpected = df.filter(
            ~F.col(COL_PRED_LABEL).isin([0.0, 1.0, 2.0, 3.0])
        ).count()
        if unexpected:
            log.warning(
                "[%s] %d rows with unexpected pred_label values.", 
                stage, 
                unexpected
            )

    if not ok and raise_on_error:
        raise ValueError(f"[{stage}] Validation failed.")
    return ok
