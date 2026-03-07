"""Write Spark DataFrame to HDFS as Parquet, partitioned by year"""

from pyspark.sql import DataFrame, functions as F

from src.utils.constants import COL_INGESTION_TS, COL_YEAR
from src.utils.logger import logging

log = logging.getLogger(__name__)

def write_partition(
    df: DataFrame, 
    base_path: str,
    year: int, 
    partition_cols: list = None,
    mode: str = "overwrite",
    compression: str = "snappy"
):
    """Write a single year's DataFrame to Parquet partition"""

    if partition_cols is None:
        partition_cols = [COL_YEAR]
    
    df = df.withColumn(COL_INGESTION_TS, F.current_timestamp())
    if COL_YEAR not in df.columns:
        df = df.withColumn(COL_YEAR, F.lit(year))
    
    log.info(
        "Writing year=%d partition to %s (mode=%s, compression=%s)",
        year, base_path, mode, compression
    )

    (
        df.write
        .option("compression", compression)
        .option("partitionOverwriteMode", "dynamic")
        .partitionBy(*partition_cols)
        .mode(mode)
        .parquet(base_path)
    )

    log.info("Successfully wrote year=%d to %s", year, base_path)
