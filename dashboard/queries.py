"""Data loading and query helpers for Streamlit dashboard."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import (
    COL_METRIC_NAME,
    COL_METRIC_VALUE,
    COL_MODEL_NAME,
    COL_PRED_LABEL,
    COL_PROB_0,
    COL_PROB_1,
    COL_YEAR,
    INFERENCE_YEARS,
    TRAIN_YEAR
)

def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a Parquet file, return warning if missing"""

    if not path.exists():
        return None
    
    return pd.read_parquet(path)

def _wilson_ci(success: int, n: int) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""

    if not n:
        return 0.0, 0.0
    
    z = 1.96
    p = success / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)

def load_predictions(base_path: str) -> pd.DataFrame:
    """Load and concatenate per year exports into single dataframe"""

    frames = []
    years = [TRAIN_YEAR] + INFERENCE_YEARS

    for year in years:
        path = Path(base_path) / f"predictions_{year}.parquet"
        df = _read_parquet(path)

        if df is None:
            continue
        if COL_YEAR not in df.columns:
            df[COL_YEAR] = year
        
        frames.append(df)
    
    if not frames:
        raise FileNotFoundError(
            f"No prediction exports found in '{base_path}'. "
            "Run the pipeline with --task dashboard first."
        )
    
    out_df = pd.concat(frames, ignore_index=True)
    out_df[COL_YEAR] = out_df[COL_YEAR].astype(int)
    out_df[COL_PRED_LABEL] = pd.to_numeric(out_df[COL_PRED_LABEL], errors="coerce")

    for col in (COL_PROB_0, COL_PROB_1):
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
    
    return out_df

def load_drift_metrics(base_path: str, metric_filter: Optional[str] = None):
    """Load drift metrics export"""

    df_path = Path(base_path) / "drift_metrics.parquet"
    df = _read_parquet(path=df_path)

    if df is None:
        raise FileNotFoundError(f"drift_metrics.parquet not found in {base_path}")
    
    df[COL_YEAR] = df[COL_YEAR].astype(int)
    df[COL_METRIC_VALUE] = pd.to_numeric(df[COL_METRIC_VALUE], errors="coerce")

    if metric_filter:
        df = df[df[COL_METRIC_NAME].str.contains(metric_filter, case=False, na=False)]
    
    return df

def load_year_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predictions into summary using partisanship trend chart"""

    rows = []
    predictions = predictions_df.groupby([COL_YEAR, COL_MODEL_NAME])
    
    for (year, model), group in predictions:
        total = len(group)
        partisan = (group[COL_PRED_LABEL] == 1.0).sum()
        frac = partisan / total if total > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci(partisan, total)
        rows.append({
            COL_YEAR: year,
            COL_MODEL_NAME: model,
            "partisan_fraction": round(frac, 4),
            "total_paragraphs": total,
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4)
        })
    
    out_df = pd.DataFrame(rows).sort_values([COL_MODEL_NAME, COL_YEAR])
    return out_df
