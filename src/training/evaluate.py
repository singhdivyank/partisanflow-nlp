"""Model evaluation: compute classification metrics"""

import numpy as np
from typing import List

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql import DataFrame, functions as F, types as T

from src.utils.constants import (
    COL_PREDICTION, 
    COL_PROBABILITY, 
    COL_RAW_LABEL, 
    COL_RAW_PRED
)
from src.utils.logger import logging

log = logging.getLogger(__name__)

def _multiclass_metrics(predictions: DataFrame) -> dict:
    metrics = {}
    evaluator = MulticlassClassificationEvaluator(
        label_col=COL_RAW_LABEL,
        predictionCol=COL_PREDICTION
    )

    for metric_name in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        metrics[metric_name]=evaluator.evaluate(
            predictions, 
            {evaluator.metricName: metric_name}
        )
    
    return metrics

def _binary_metrics(predictions: DataFrame) -> dict:
    metrics = {}
    evaluator = BinaryClassificationEvaluator(
        labelCol=COL_RAW_LABEL,
        rawPredictionCol=COL_RAW_PRED
    )

    for metric_name in ["areaUnderROC", "areaUnderPR"]:
        try:
            metrics[metric_name] = evaluator.evaluate(
                predictions, 
                {evaluator.metricName: metric_name}
            )
        except Exception as e:
            log.warning("Could not compute binary metric '%s': '%s'", metric_name, e)
    
    return metrics

def _collect_arrays(predictions: DataFrame) -> List[np.array]:
    """Collect y_true, y_pred, and probability scores to driver"""

    y_true = predictions.select(COL_RAW_LABEL).rdd.flatMap(lambda x: x).collect()
    y_pred = predictions.select(COL_PREDICTION).rdd.flatMap(lambda x: x).collect()
    y_scores = None

    if COL_PROBABILITY in predictions.columns:
        v2arr = F.udf(lambda v: v.toArray().toList(), T.ArrayType(T.DoubleType()))
        prob_df = predictions.withColumn("_prob_arr", v2arr(F.col(COL_PROBABILITY)))
        rows = prob_df.select("_prob_arr").toPandas()
        y_scores = np.array(rows["_prob_arr"].tolist())
    
    return [np.array(y_true), np.array(y_pred), y_scores]

def evaluate_model(
    predictions: DataFrame, 
    model_name: str, 
    has_probability: bool = True
) -> dict:
    """Compute all evaluation metrics, generate plots, and log artifacts"""

    log.info("Evaluate model: %s", model_name)

    metrics = _multiclass_metrics(predictions)
    if has_probability:
        binary = _binary_metrics(predictions)
        metrics.update(binary)
    
    for k, v in metrics.items():
        log.info("%-25s %.4f", k, v)
    
    return metrics
