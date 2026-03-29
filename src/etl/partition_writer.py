"""Write Spark DataFrame to HDFS as Parquet, partitioned by year"""

from pyspark.sql import (
    DataFrame,
    DataFrameWriter,
    SparkSession, 
    functions as F, 
)

from src.utils.constants import COL_INGESTION_TS, COL_YEAR
from src.utils.logger import get_logger

log = get_logger(__name__)

def write_partition(
    df: DataFrame, 
    base_path: str,
    year: int, 
    ts_col: str = COL_INGESTION_TS,
    mode: str = "overwrite",
    compression: str = "snappy"
):
    """Write a single year's DataFrame to Parquet partition"""
    
    df = df.withColumn(ts_col, F.current_timestamp())\
        .withColumn(COL_YEAR, F.lit(year))\
        .repartition(F.col(COL_YEAR))
    
    log.info(
        "Writing year=%d partition to %s (mode=%s, compression=%s)",
        year, 
        base_path, 
        mode, 
        compression
    )

    (
        df.write
        .option("compression", compression)
        .option("parquet.block.size", 256 * 1024 * 1024)
        .mode(mode)
        .partitionBy(COL_YEAR)
        .parquet(base_path)
    )

    log.info("Successfully wrote year=%d to %s", year, base_path)

def write_to_delta(
    writer: DataFrameWriter,
    path: str,
    overwrite_schema: bool = False
) -> None:
    """Write joined dataframe to Delta Lake table"""
    
    writer = (
        writer.format("delta")
        .option("optimizeWrite", "true")
        .option("autoCompact", "true")
        .option("delta.targetFileSize", "256MB")
        .partitionBy(*COL_YEAR)
    )

    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    
    writer.mode("append").save(path)

def read_partition(
    spark: SparkSession, 
    base_path: str, 
    year: int
) -> DataFrame:
    """Read specific year partition from Parquet store"""

    path = f"{base_path}/year={year}"
    
    log.info("Reading partition from %s", path)
    
    df = spark.read.parquet(path)
    
    log.info(
        "Loaded partition for year=%d from %s", 
        year,
        path
    )
    
    return df
