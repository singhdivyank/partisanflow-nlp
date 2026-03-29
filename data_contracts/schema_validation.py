"""Data contract validation for every stage of pipeline."""

from pyspark.sql import DataFrame, functions as F

from validation_utils import (
    _load_expectations,
    _check_required_columns,
    _check_label_distribution,
    _check_row_count
)
from src.utils.constants import (
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

    min_rows = exp.get("min_row_count", 1)
    n = df.count()
    if not _check_row_count(n, min_rows, stage):
        ok = False

    log.info("Completed row count check")

    if not ok and raise_on_error:
        raise ValueError(f"[{stage}] Validation failed. See logs for details.")
    
    return ok

def validate_year(df: DataFrame, year: int, is_training: bool = False) -> bool:
    """Orchestrate quality checks for a year"""

    log.info("Quality checks on preprocessed data...")
    ok = True
    
    if is_training and COL_RAW_LABEL in df.columns:
        present_labels = {
            row[COL_RAW_LABEL]
            for row in df.select(COL_RAW_LABEL).dropna().distinct().collect()
        }
        
        missing = [lbl for lbl in [0, 1] if lbl not in present_labels]
        if missing:
            log.error(
                "[validate | year=%d] Training labels %s not found in data.", 
                year, 
                missing
            )
            ok = False
        else:
            log.info("[validate | year=%d] All training labels present.", year)
            ok = False
    elif is_training:
        log.error("[validate | year=%d] All training labels missing", year)
        ok = False

    if not ok:
        raise ValueError(f"Data validation failed for year={year}. See logs for details.")

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
