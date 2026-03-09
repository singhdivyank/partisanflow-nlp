"""
Airflow DAG: Orchestrates the full ML pipeline from ETL through drift monitoring.
All tasks are idempotent and use SparkSubmitOperator for cluster execution.
"""

from datetime import datetime, timedelta
from pathlib import Path

import yaml

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.operators.empty import EmptyOperator

DAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAG_DIR.parent

airflow_config = PROJECT_ROOT / "config" / "airflow_config.yaml"
base_config = PROJECT_ROOT / "config" / "base_config.yaml"

with open(airflow_config) as f:
    _CFG = yaml.safe_load(f)

with open(base_config) as f:
    _BASE = yaml.safe_load(f)

_DAG_CFG = _CFG["dag"]
_ARGS_CFG = _CFG["default_args"]
_SPARK_CFG = _CFG["spark_submit"]
_TIMEOUTS = _CFG["timeouts"]
INFERENCE_YEARS = _BASE["years"]["inference"]

default_args = {
    "owner": _ARGS_CFG["owner"],
    "retries": _ARGS_CFG["retries"],
    "retry_delay": timedelta(minutes=_ARGS_CFG["retry_delay_min"]),
    "email_on_failure": _ARGS_CFG["email_on_failure"],
    "depends_on_past": False,
}

def _spark_task(
    task_id: str,
    task_args: str,
    timeout_min: int,
    extra_args: list = None
) -> SparkSubmitOperator:
    """Factory function"""

    return SparkSubmitOperator(
        conn_id=_SPARK_CFG["conn_id"],
        application=_SPARK_CFG["application"],
        deploy_mode=_SPARK_CFG["deploy_mode"],
        driver_memory=_SPARK_CFG["driver_memory"],
        executor_memory=_SPARK_CFG["executor_memory"],
        num_executors=_SPARK_CFG["num_executors"],
        executor_cores=_SPARK_CFG["executor_cores"],
        application_args=["--task", task_args] + (extra_args or []),
        execution_timeout=timedelta(minutes=timeout_min),
        task_id=task_id
    )

with DAG(
    dag_id=_DAG_CFG["dag_id"],
    description=_DAG_CFG["description"],
    schedule=_DAG_CFG["schedule"],
    start_date=datetime.strptime(_DAG_CFG["start_date"], "%Y-%m-%d"),
    catchup=_DAG_CFG["catchup"],
    max_active_runs=_DAG_CFG["max_active_runs"],
    tags=_DAG_CFG["tags"],
    default_args=default_args,
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    etl = _spark_task(
        task_id="etl_1869",
        task_args="etl",
        timeout_min=_TIMEOUTS["etl"],
        extra_args=["--year", "1869"]
    )
    features = _spark_task(
        task_id="features_1869",
        task_args="features",
        timeout_min=_TIMEOUTS["features"],
        extra_args=["--year", "1869"]
    )
    train_models = _spark_task(
        task_id="train_models_1869",
        task_args="train",
        timeout_min=_TIMEOUTS["train"],
    )
    register_model = _spark_task(
        task_id="register_best_model",
        task_args="register",
        timeout_min=_TIMEOUTS["register"],
    )

    # per year inference
    prev_inference = register_model
    inference_ends = []

    for year in INFERENCE_YEARS:
        etl_year = _spark_task(
            task_id=f"etl_{year}",
            task_args="etl",
            timeout_min=_TIMEOUTS["etl"],
            extra_args=["--year", str(year)]
        )
        features_year = _spark_task(
            task_id=f"features_{year}",
            task_args="features",
            timeout_min=_TIMEOUTS["features"],
            extra_args=["--year", str(year)]
        )
        predict_year = _spark_task(
            task_id=f"batch_predict_{year}",
            task_args="predict",
            timeout_min=_TIMEOUTS["inference"],
            extra_args=["--year", str(year)]
        )
        drift_year = _spark_task(
            task_id=f"compute_drift_{year}",
            task_args="drift",
            timeout_min=_TIMEOUTS["drift"],
            extra_args=["--year", str(year)]
        )

        register_model >> etl_year >> features_year >> predict_year >> drift_year
        inference_ends.append(drift_year)
    
    # dashboard
    refresh_dashboard = _spark_task(
        task_id="refresh_dashboard_views",
        task_args="dashboard",
        timeout_min=_TIMEOUTS["dashboard"],
    )

    # wire top-level dependencies
    start >> etl >> features >> train_models >> register_model
    inference_ends >> refresh_dashboard >> end
