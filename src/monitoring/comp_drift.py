"""Data and concept drift computation. Uses 1869 predictions as reference distribution."""

import numpy as np

from pyspark.sql import DataFrame, SparkSession, functions as F

from .helpers import (
    _comp_drift_score, 
    _class_balance, 
    _label_flip_rate,
    _prob_summary_stats, 
    _row
)
from src.utils.constants import (
    COL_PROB_1,
    COL_YEAR,
    DRIFT_KL_PROB1,
    DRIFT_PSI_PROB1,
    MLFLOW_REGISTRY_NAME, 
    PSI_CRITICAL_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    REFERENCE_YEAR
)
from src.utils.logger import logging

log = logging.getLogger(__name__)

def compute_data_drift(
    ref_df: DataFrame,
    cur_df: DataFrame,
    year: int,
    model_version: str
) -> list[dict]:
    """
    Compute drifft metrics comparing `cur_df` against `ref_df` over a year.
    Tracks shift in input data.
    """

    log.info(
        "Computing drift: year=%d vs ref=%d, model=%s", 
        year, REFERENCE_YEAR, MLFLOW_REGISTRY_NAME
    )

    ref_arr = np.array(
        ref_df.select(COL_PROB_1)
        .filter(F.col(COL_PROB_1).isNotNull())
        .rdd.flatMap(lambda x: x).collect()
    )
    cur_arr = np.array(
        cur_df.select(COL_PROB_1)
        .filter(F.col(COL_PROB_1).isNotNull())
        .rdd.flatMap(lambda x: x).collect()
    )

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        log.warning("Insufficient data for drift computation (year=%d).", year)
        return []
    
    rows = []
    psi = _comp_drift_score(ref_arr, cur_arr, score_name="psi")
    rows.append(_row(DRIFT_PSI_PROB1, psi, year, model_version))
    kl_score = _comp_drift_score(ref_arr, cur_arr, score_name="kl")
    rows.append(_row(DRIFT_KL_PROB1, kl_score, year, model_version))

    if psi >= PSI_CRITICAL_THRESHOLD:
        log.warning(
            "CRITICAL DRIFT | year=%d | model=%s | PSI=%.4f (>= %.2f)",
            year, MLFLOW_REGISTRY_NAME, psi, PSI_CRITICAL_THRESHOLD
        )
    
    if psi >= PSI_WARNING_THRESHOLD:
        log.warning(
            "MODERATE DRIFT | year=%d | model=%s | PSI=%.4f (>= %.2f)",
            year, MLFLOW_REGISTRY_NAME, psi, PSI_WARNING_THRESHOLD
        )

    return rows

def compute_concept_drift(
    ref_df: DataFrame,
    cur_df: DataFrame,
    year: int,
    model_version: str
) -> list[dict]:
    """
    Compute all concept drift metrics for single inference year.
    Tracks shifts in model behaviour.
    """

    log.info(
        "Computing concept drift: year=%d vs ref=%d, model=%s",
        year, REFERENCE_YEAR, MLFLOW_REGISTRY_NAME
    )

    metrics = []

    class_stats = _class_balance(cur_df, year, model_version)
    if class_stats:
        log.info(
            "[concept_drift | year=%d] Predicted class distribution: %s",
            year,
            {r['label']: r['frac'] for r in class_stats},
        )
    else:
        log.info("[concept_drift | year=%d] No predicted class distribution", year)
    metrics += class_stats

    prob_stats = _prob_summary_stats(cur_df, year, model_version)
    if not prob_stats:
        log.warning("[concept_drift | year=%d] No prob_1 or prob_0 values found.", year)
    metrics += prob_stats

    label_flips = _label_flip_rate(ref_df, cur_df, year, model_version)
    if not label_flips:
        log.warning(
            "[concept_drift | year=%d] No overlapping paragraphs for flip-rate computation.",
            year
        )
    else:
        log.info(
            "[concept_drift | year=%d] Label flip rate vs %d: %.4f",
            year, REFERENCE_YEAR, label_flips['flip_rate']
        )
    metrics += label_flips

    log.info(
        "[concept_drift | year=%d] %d metric rows computed", year, len(metrics)
    )

def write_drift_metrics(
    spark: SparkSession,
    metrics: list[dict],
    paths: str
) -> None:
    """Persist drift metrics to table, rows are overwritten"""

    if not metrics:
        log.warning("No drift metrics to write")
        return
    
    df = spark.createDataFrame(metrics)
    (
        df.write
        .option("compression", "snappy")
        .partitionBy(COL_YEAR)
        .mode("overwrite")
        .parquet(paths)
    )

    log.info("Drift metrics written to %s (%d rows)", paths, len(metrics))
