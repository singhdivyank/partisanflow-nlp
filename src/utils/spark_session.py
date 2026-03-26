"""
Centralised Spark Session Factory. 
All pipeline modues obtain session via get_spark_session().
"""

import os
from typing import Optional

import yaml
from pyspark.sql import SparkSession

from .constants import SPARK_CONFIG_PATH
from .logger import get_logger

log = get_logger(__name__)


def _load_spark_cfg() -> dict:
    with open(SPARK_CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_spark_session(
    app_name: Optional[str] = None,
    extra_conf: Optional[dict] = None,
) -> SparkSession:
    """
    Build or retrieve an existing SparkSession.

    Parameters
    ----------
    app_name : str, optional
        Overrides the app name from spark_config.yaml.
    extra_conf : dict, optional
        Additional Spark conf key-value pairs that take precedence
        over values in spark_config.yaml.

    Returns
    -------
    SparkSession
    """
    cfg = _load_spark_cfg()

    name = app_name or cfg.get("app_name", "newspaper-partisanship-ml")
    # master = cfg.get("master", "yarn")

    builder = SparkSession.builder.appName(name)

    # Apply config from YAML
    for k, v in cfg.get("spark_conf", {}).items():
        builder = builder.config(k, str(v))

    # Apply Hadoop config
    for k, v in cfg.get("hadoop_conf", {}).items():
        builder = builder.config(f"spark.hadoop.{k}", str(v))

    # Caller overrides (highest priority)
    for k, v in (extra_conf or {}).items():
        builder = builder.config(k, str(v))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    log.info("SparkSession created: app=%s", name)
    return spark

def stop_spark_session(spark: SparkSession) -> None:
    """Gracefully stop the SparkSession."""
    if spark:
        log.info("Stopping SparkSession.")
        spark.stop()
