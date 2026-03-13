"""
Model training endpoint. 
Builds Spark ML classifier pipelines, perform train_test split, trains all enabled models.
"""

from typing import Any, Union

import mlflow

from pyspark.sql import DataFrame

from .create_plots import (
    plot_confusion_matrix, 
    plot_pr, 
    plot_roc
)
from .evaluate import evaluate_model, _collect_arrays
from .helper import _BUILDER_MAP
from src.utils.constants import ALL_MODELS, COL_SERIES, COL_ISSUE
from src.utils.logger import logging

log = logging.getLogger(__name__)

def _log_info(model_name: str, run_id: Any, param_items: list):
    log.info("Training model=%s run_id=%s", model_name, run_id)

    mlflow.log_param("model_type", model_name)
    mlflow.log_param("feature_store", "newspaper_features_v1")
    for k, v in param_items:
        mlflow.log_param(k, v)

def plot_curves(predictions: DataFrame, has_probability: bool, model_name: str):
    _collected_arrays = _collect_arrays(predictions)
    y_true, y_pred, y_scores = _collected_arrays[0], _collected_arrays[1], _collected_arrays[-1]
    cm_path = plot_confusion_matrix(y_true, y_pred, model_name)
    mlflow.log_artifact(str(cm_path))

    if has_probability and y_scores is not None:
        num_cls = y_scores.shape[1]
        roc_path = plot_roc(y_true, y_scores, model_name, num_cls)
        pr_path = plot_pr(y_true, y_scores, model_name, num_cls)
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(pr_path))

def train_all_models(
    model_cfg: dict, 
    train_df: DataFrame, 
    test_df: DataFrame
):
    """Train all enabled models and return dict of model properties"""

    results = {}
    model, eval_metrics, run_id = None, None, None

    mlflow_cfg = model_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name")
    
    for model_name in ALL_MODELS:
        model_config = model_cfg["models"][model_name]
        
        if not model_config.get("enabled", True):
            log.info("Model '%s' is disabled in config — skipping", model_name)
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        params = model_config.get("params", {})
        classifier = _BUILDER_MAP[model_name](params)
        
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            _log_info(model_name=model_name, run_id=run_id, param_items=params.items())
            
            # fit the model
            model = classifier.fit(train_df)
            # make predictions
            predictions = model.transform(test_df)
            has_probability=(model_name != ALL_MODELS[-1])
            eval_metrics = evaluate_model(
                predictions=predictions, 
                model_name=model_name,
                has_probability=has_probability
            )
            
            plot_curves(predictions, has_probability, model_name)
            
            for metric_name, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metrics(metric_name, value)
            
            mlflow.spark.log_model(model, artifact_path="model")
        
        if model is not None:
            results[model_name] = {"model": model, "metrics": eval_metrics, "run_id": run_id}
            log.info(
                "Model=%s  accuracy=%.4f  f1=%.4f",
                model_name,
                eval_metrics.get("accuracy", 0),
                eval_metrics.get("f1", 0),
            )
    
    return results

def perform_split(df: DataFrame, split_cfg: dict) -> Union[DataFrame, DataFrame]:
    """
    Group-aware train test split: all paragraphs belonging to 
    same issue end up in same split to prevent data leakage.
    """
    
    series_df = df.select(COL_ISSUE).distinct()
    series_train, series_test = series_df.randomSplit(
        weights=[split_cfg.get("train_ratio"), split_cfg.get("test_ratio")],
        seed=split_cfg.get("seed")
    )

    train_df = df.join(series_train, on=COL_SERIES, how="inner")
    test_df = df.join(series_test, on=COL_SERIES, how="inner")
    
    log.info(
        "Split — train rows: %d, test rows: %d", 
        train_df.count(), test_df.count()
    )

    return train_df, test_df
