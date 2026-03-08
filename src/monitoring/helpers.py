import numpy as np
from datetime import datetime, UTC

from pyspark.sql import DataFrame, functions as F

from src.utils.constants import (
    COL_COMPUTED_TS,
    COL_ISSUE,
    COL_MODEL_NAME,
    COL_MODEL_VERSION,
    COL_METRIC_NAME,
    COL_METRIC_VALUE,
    COL_PARAGRAPH_ID,
    COL_PRED_LABEL,
    COL_PROB_0,
    COL_PROB_1,
    COL_REF_YEAR,
    COL_SERIES,
    COL_YEAR,
    DRIFT_HIGH_CONF_FRAC,
    DRIFT_MEAN_PROB0,
    DRIFT_MEAN_PROB1,
    DRIFT_VAR_PROB0,
    DRIFT_VAR_PROB1,
    HIGH_CONF_PROB_THRESHOLD,
    REFERENCE_YEAR,
    MLFLOW_REGISTRY_NAME
)

def _row(
    name: str, 
    value: float, 
    year: int, 
    model_version: str
) -> dict:
    
    return {
        COL_YEAR: year,
        COL_REF_YEAR: REFERENCE_YEAR,
        COL_MODEL_NAME: MLFLOW_REGISTRY_NAME,
        COL_MODEL_VERSION: model_version,
        COL_METRIC_NAME: name,
        COL_METRIC_VALUE: value,
        COL_COMPUTED_TS: datetime.now(UTC).isoformat(),
    }


# DATA DRIFT HELPER FUNCTIONS

def _comp_drift_score(
    ref_arr: np.ndarray, 
    cur_arr: np.ndarray, 
    score_name: str
) -> float:
    """Functions for computing psi and kl-divergence scores"""

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    ref_counts, _ = np.histogram(ref_arr, bins=bins)
    cur_counts, _ = np.histogram(cur_arr, bins=bins)

    if score_name == "psi":
        # psi computation
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (len(ref_arr) + eps * n_bins)
        cur_pct = (cur_counts + eps) / (len(cur_arr) + eps * n_bins)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi, 6)
    else:
        # kl divergence
        eps = 1e-10
        ref_p = (ref_counts + eps) / (ref_counts + eps).sum()
        cur_p = (cur_counts + eps) / (cur_counts + eps).sum()
        kl = float(np.sum(ref_p * np.log(ref_p / cur_p)))
        return round(kl, 6)


# CONCEPT DRFIT HELPER FUNCTIONS

def _class_balance(
    cur_df: DataFrame, 
    year: int, 
    model_version: str
) -> list[dict]:
    """Fraction of paragraphs predicted as each class."""

    total = cur_df.count()
    if not total:
        return []
    
    rows = []
    dist = (
        cur_df.groupBy(COL_PRED_LABEL)
        .count()
        .collect()
    )

    for r in dist:
        label = int(r[COL_PRED_LABEL])
        frac = r["count"] / total
        row_dict = _row(
            name=f"pred_class_{label}_fraction", 
            value=frac, 
            year=year, 
            model_version=model_version
        )
        rows.append(row_dict)
    
    return rows

def _prob_summary_stats(
    cur_df: DataFrame,
    year: int,
    model_version: str
) -> list[dict]:
    """Mean, variance, and high-confidence fraction of prob_1 and prob_0"""

    prob1 = (
        cur_df.select(COL_PROB_1)
        .filter(F.col(COL_PROB_1).isNotNull())
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    prob0 = (
        cur_df.select(COL_PROB_0)
        .filter(F.col(COL_PROB_0).isNotNull())
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    arr_prob1 = np.arr(prob1)
    arr_prob0 = np.arr(prob0)

    if not arr_prob1.size:
        return []
    
    if not arr_prob0.size:
        return []
    
    high_conf_0 = float(np.mean((arr_prob0 > HIGH_CONF_PROB_THRESHOLD) | (arr_prob0 < 1 - HIGH_CONF_PROB_THRESHOLD)))
    high_conf_1 = float(np.mean((arr_prob1 > HIGH_CONF_PROB_THRESHOLD) | (arr_prob1 < 1 - HIGH_CONF_PROB_THRESHOLD)))

    return [
        _row(name=DRIFT_MEAN_PROB0, value=arr_prob0.mean(), year=year, model_version=model_version),
        _row(name=DRIFT_MEAN_PROB1, value=arr_prob1.mean(), year=year, model_version=model_version),
        _row(name=DRIFT_VAR_PROB0, value=arr_prob0.var(), year=year, model_version=model_version),
        _row(name=DRIFT_VAR_PROB1, value=arr_prob1.var(), year=year, model_version=model_version),
        _row(name=DRIFT_VAR_PROB1, value=arr_prob1.var(), year=year, model_version=model_version),
        _row(name=f"{DRIFT_HIGH_CONF_FRAC}_democrat", value=high_conf_0, year=year, model_version=model_version),
        _row(name=f"{DRIFT_HIGH_CONF_FRAC}_republican", value=high_conf_1, year=year, model_version=model_version),
    ]
    
def _label_flip_rate(
    ref_df: DataFrame,
    cur_df: DataFrame,
    year: int,
    model_version: str
) -> list[dict]:
    """Fraction of paragraphs that have a different predicted label in `cur_df` vs `ref_df`"""

    JOIN_KEYS = [COL_SERIES, COL_ISSUE, COL_PARAGRAPH_ID]
    ref_preds = ref_df.select(
        *JOIN_KEYS, F.col(COL_PRED_LABEL).alias("ref_pred")
    )
    cur_preds = cur_df.select(
        *JOIN_KEYS, F.col(COL_PRED_LABEL).alias("cur_pred")
    )

    joined = ref_preds.join(cur_preds, on=JOIN_KEYS, how="inner")
    total = joined.count()

    if not total:
        return []
    
    flipped = joined.filter(F.col("ref_pred") != F.col("cur_pred")).count()
    return [_row(name="label_flip_rate", value=flipped / total, year=year, model_version=model_version)]
