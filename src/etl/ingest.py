"""Read raw parquet and metadata CSV, joins on series_id, and attach labels for training year."""

from pyspark.sql import DataFrame, SparkSession, functions as F

from src.utils.constants import (
    COL_CONTENTS,
    COL_DATE,
    COL_ISSUE,
    COL_SERIES,
    COL_TEXT,
    COL_RAW_LABEL,
    EXCLUDED_SERIES,
    TRAIN_YEAR,
)
from src.utils.logger import get_logger

log = get_logger()

def read_parquet(spark: SparkSession, path: str, year: int) -> DataFrame:
    """
    Read the raw newspaper Parquet files and filter to a single year.

    Parameters
    ----------
    spark : SparkSession
    path  : str   Base HDFS path (year=* glob or explicit partition path).
    year  : int   Calendar year to filter on.

    Returns
    -------
    DataFrame with columns: series_id, issue_id, date, text
    """

    log.info("Reading raw Parquet from %s, filtering year=%d", path, year)
    
    df = (
        spark.read.parquet(path)
        .filter(F.year(F.col(COL_DATE))==1869)
        .select(COL_SERIES, COL_ISSUE, COL_DATE, COL_TEXT)
    )
    
    log.info("Raw Parquet rows for year=%d: %d", year, df.count())
    return df

def read_metadata(
    spark: SparkSession, 
    path: str,
    label_col: str, 
    delimiter: str,
    labels: dict
) -> DataFrame:
    """
    Read the metadata CSV and assign numeric party labels.

    Labels are derived from semicolon-delimited values in the `contents`
    column:
        democratic  → 0.0
        republican  → 1.0
        independent → 2.0
        other       → 3.0

    Rows with series in EXCLUDED_SERIES and null series are dropped.

    Returns
    -------
    DataFrame with columns: series_id, metadata (alias of contents), issue_label
    """

    log.info("Reading metadata CSV from %s", path)

    df = (
        spark.read.csv(path, header=True, inferSchema=True)
        .select(COL_CONTENTS, COL_SERIES)
        .dropna(subset=[COL_SERIES])
        .filter(~F.col(COL_SERIES).isin(EXCLUDED_SERIES))
        .filter(~F.col(COL_SERIES).isin(EXCLUDED_SERIES))
    )

    contents_arr = F.split(F.col(label_col), delimiter)
    df = df.withColumn(
        COL_RAW_LABEL,
        F.when(F.array_contains(contents_arr, "democratic"), labels['democratic'])
        .when(F.array_contains(contents_arr, "republican"), labels['republican'])
        .when(F.array_contains(contents_arr, "independent"), labels['independent'])
        .otherwise(labels['other']),
    )

    df = df.select(
        COL_SERIES,
        F.col(COL_CONTENTS).alias("metadata"),
        F.col(COL_RAW_LABEL).alias(COL_RAW_LABEL),
    )

    log.info("Metadata rows loaded: %d", df.count())
    return df

def ingest(
    spark: SparkSession,
    parquet_path: str,
    metadata_path: str,
    label_col: str,
    label_delimiter: str,
    labels: dict,
    year: int = TRAIN_YEAR
) -> DataFrame:
    """
    Full ingestion pipeline for a given year.
    Returns a labelled DataFrame ready for the cleaning / transform stage.
    """

    raw_df = read_parquet(spark=spark, path=parquet_path, year=year)
    metadata_df = read_metadata(
        spark=spark, 
        path=metadata_path, 
        label_col=label_col, 
        delimiter=label_delimiter,
        labels=labels
    )
    
    log.info("Joining raw data with metadata on '%s'", COL_SERIES)

    joined_df = raw_df.join(metadata_df, on=COL_SERIES, how="inner")
    joined_df = joined_df.filter(joined_df["label"].isin(labels["train_labels"]))

    log.info("Rows after inner join: %d", joined_df.count())
    return joined_df
