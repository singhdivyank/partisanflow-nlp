import json
from pathlib import Path

from pyspark.sql import DataFrame, types as T
from pyspark.ml.linalg import VectorUDT

from src.utils.constants import (
    COL_CONTENTS, 
    COL_DATE,
    COL_FEATURES,
    COL_ISSUE,
    COL_RAW_LABEL,
    COL_SERIES,
    COL_TEXT,
    COL_YEAR,
    COL_PARAGRAPH_ID
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_EXPECTATIONS_PATH = Path(__file__).parent / "expectations.json"

RAW_PARQUET_SCHEMA = T.StructType([
    T.StructField(COL_SERIES, T.StringType(),    nullable=False),
    T.StructField(COL_ISSUE,  T.StringType(),    nullable=False),
    T.StructField(COL_DATE,   T.DateType(),      nullable=True),
    T.StructField(COL_TEXT,   T.StringType(),    nullable=False),
])

METADATA_SCHEMA = T.StructType([
    T.StructField(COL_SERIES,   T.StringType(), nullable=False),
    T.StructField(COL_CONTENTS,   T.StringType(), nullable=True),
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
