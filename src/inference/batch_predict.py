"""
Batch inference pipeline. Loads the production model from MLflow, 
runs predictions, and returns standardised prediction dataframe.
"""

import mlflow
from typing import Union

from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame, functions as F, types as T

from src.utils.constants import (
    COL_MODEL_NAME,
    COL_MODEL_VERSION,
    COL_ISSUE,
    COL_PARAGRAPH_ID,
    COL_PROB_0,
    COL_PROB_1,
    COL_PREDICTION,
    COL_PRED_LABEL,
    COL_PRED_TS,
    COL_PROBABILITY,
    COL_RAW_LABEL,
    COL_SERIES,
    COL_YEAR,
    MLFLOW_REGISTRY_NAME,
    MLFLOW_STAGE_PRODUCTION
)
from src.utils.logger import logging

log = logging.getLogger(__name__)

OUTPUT_COLS = [
    COL_SERIES, COL_ISSUE, COL_PARAGRAPH_ID, COL_YEAR,
    COL_PRED_LABEL, COL_PROB_0, COL_PROB_1,
    COL_MODEL_NAME, COL_MODEL_VERSION, COL_PRED_TS
]

_extract_prob = F.udf(
    lambda v, i: float(v[i]) if v is not None and len(v) > i else None,
    T.DoubleType()
)

def load_production_model(tracking_uri: str) -> Union[PipelineModel, str]:
    """Load current producion model from MLflow model registry"""

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MLFLOW_REGISTRY_NAME}/{MLFLOW_STAGE_PRODUCTION}"
    log.info("Loading production model from %s", model_uri)
    model = mlflow.spark.load_model(model_uri)
    log.info("Production model loaded")

    client = mlflow.tracking.MlflowClient()
    prod_version = client.get_latest_versions(MLFLOW_REGISTRY_NAME, stages=[MLFLOW_STAGE_PRODUCTION])
    version = prod_version[0].version if prod_version else "unknown"
    return model, version

def run_predictions(
    model: PipelineModel,
    feature_df: DataFrame,
    model_version: str,
    year: int,
    has_probability: bool = True
) -> DataFrame:
    """Run batch predictions on feature dataframe"""

    log.info("Running batch predictions for year=%d", year)

    preds = model.transform(feature_df)
    preds = preds.withColumnRenamed(COL_PREDICTION, COL_PRED_LABEL)

    if has_probability and COL_PROBABILITY in preds.columns:
        preds = (
            preds
            .withColumn(COL_PROB_0, _extract_prob(F.col(COL_PROBABILITY), F.lit(0)))
            .withColumn(COL_PROB_1, _extract_prob(F.col(COL_PROBABILITY), F.lit(1)))
        )
    else:
        preds = (
            preds
            .withColumn(COL_PROB_0, F.lit(None).cast(T.DoubleType()))
            .withColumn(COL_PROB_1, F.lit(None).cast(T.DoubleType()))
        )
    
    preds = (
        preds
        .withColumn(COL_MODEL_NAME, F.lit(MLFLOW_REGISTRY_NAME))
        .withColumn(COL_MODEL_VERSION, F.lit(model_version))
        .withColumn(COL_PRED_TS, F.current_timestamp())
    )

    if COL_RAW_LABEL in preds.columns:
        OUTPUT_COLS.insert(4, COL_RAW_LABEL)
    
    preds = preds.select(*OUTPUT_COLS)
    log.info("Predictions generated for year=%d: %d rows", year, preds.count())
    return preds
