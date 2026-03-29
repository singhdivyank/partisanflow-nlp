"""
Centralised Spark Session Factory with Delta Lake Support. 
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
    enable_delta: bool = True
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
    builder = SparkSession.builder.appName(name)
    
    # Enable Delta Lake if requested
    if enable_delta:
        log.info("Configuring spark with Delta Lake support")
        builder = builder.config(
            "spark.sql.extensions", 
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        builder = builder.config(
            "spark.sql.catalog.spark_catalog", 
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
    
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
    spark.conf.set("spark.sql.shuffle.partitions", 4)

    # Delta lake specific configurations
    if enable_delta:
        # spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
        spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "false")
        # spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
        spark.conf.set("spark.databricks.delta.autoCompact.enabled", "false")
        spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

        log.info("Delta lake extensions configured successfully")

    log.info(
        "SparkSession created: app=%s, delta_enabled=%s", 
        name, 
        enable_delta
    )
    return spark

def stop_spark_session(spark: SparkSession) -> None:
    """Gracefully stop the SparkSession."""
    if spark:
        log.info("Stopping SparkSession.")
        spark.stop()
