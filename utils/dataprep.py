"""
Functions for creating and processing training data.
"""

import glob

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def create_data(spark_session: SparkSession, parquet_loc: str) -> DataFrame:
    """
    Create initial dataframe from parquet file by splitting into paragraphs and filtering by date
    
    Parameters:
    parquet_loc (str): Location of the parquet file
    spark_session (SparkSession): Spark session object
    
    Returns:
    df: Spark DataFrame with columns series, issue, date, id, new_text
    """
    
    try:
        # read parquet file
        df = spark_session.read.parquet(parquet_loc)
        # split df into paragraphs
        df = df.withColumn("paragraphs", F.split(F.col('text'), r"\n\n+"))
        # explode into multiple rows
        df = df.withColumn("new_text", F.explode('paragraphs'))
        # drop paragraphs and text columns
        df = df.select('series', 'issue', 'date', 'id', 'new_text')
        # filter df
        df = df.filter((F.col('date')>='1869') & (F.col('date')<='1870'))
        return df
    except Exception as e:
        logger.error(f"An error occurred while creating data from {parquet_loc}: {e}")
        raise RuntimeError(f"Failed to create data: {e}")

def process_metadata(spark_session: SparkSession, csv_file: str) -> DataFrame:
    """
    Process metadata CSV to assign labels and filter out unwanted labels
    
    Parameters:
    csv_file (str): Location of the metadata CSV file
    spark_session (SparkSession): Spark session object
    
    Returns:
    DataFrame: Spark DataFrame with columns series, contents, label
    """
    
    try:
        # read metadata
        csv_df = spark_session.read.csv(csv_file, header=True, inferSchema=True).select('contents', 'series')
        # assign labels
        csv_df = csv_df.withColumn(
            "label", 
            F.when(F.lower(F.col("contents")).rlike(r"\bdemocratic\b"), 0).when(F.lower(F.col("contents")).rlike(r"\brepublican\b"), 1).when(F.lower(F.col("contents")).rlike(r"\bindependent\b"), 2).otherwise(3)
        )
        # filter out independent and neutral labels
        csv_df = csv_df.filter((F.col('label') != 3) & (F.col('label') != 2))
        # keep only relevant columns
        csv_df = csv_df.select('series', 'contents', 'label')
        return csv_df
    except Exception as e:
        logger.error(f"An error occurred while processing metadata {csv_file}: {e}")
        raise RuntimeError(f"Failed to process metadata: {e}")

def concatenate(data_dir: str) -> pd.DataFrame:
    """
    Concatenate all CSV files into a single DataFrame.

    Parameters:
    data_dir (str): Directory containing CSV files to concatenate.
    
    Returns:
    pd.DataFrame: Concatenated DataFrame of all CSV files.
    """
    
    # find all csv files in the directory
    csv_files = glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {data_dir}")
        return None
    
    try:
        first_df = pd.read_csv(csv_files[0])
        batches, temp_batch = [first_df], []

        for idx, file_name in enumerate(csv_files[1:], 1):
            # read the rest without header
            temp_df = pd.read_csv(file_name, header=None, names=first_df.columns)
            temp_batch.append(temp_df)
            if not idx % 1000:
                # concatenate batch and reset
                batch_df = pd.concat(temp_batch, ignore_index=True)
                batches.append(batch_df)
                temp_batch = []
        
        # concatenate any remaining dataframes
        if temp_batch:
            batch_df = pd.concat(temp_batch, ignore_index=True)
            batches.append(batch_df)
        
        result = pd.concat(batches, ignore_index=True)
        return result
    except Exception as e:
        logger.error(f"Error concatenating data: {e}")
        raise RuntimeError(f"Falied to concatenate data: {e}")

def create_doc_data(
        split_col: str, 
        source_file: str,
        target_file: str
    ) -> None:
    """
    Create data grouped by the specified column (issue or series) 
    combining all paragraphs into one.

    Parameters:
    split_col (str): Column name to group by ('issue' or 'series').
    source_file (str): Path to the source CSV file containing processed data.
    target_file (str): Path to save the grouped data CSV file.
    """

    feature_col = 'series' if split_col == 'issue' else 'issue'
    
    try:
        df = pd.read_csv(source_file)
        doc_df = (
            df.groupby(split_col).agg({
                'processed_text': lambda texts: ' '.join(texts), 
                feature_col: 'first', 'label': 'first'
            })
            .reset_index()
        )
        doc_df.to_csv(target_file, index=False)
        logger.info(f"Data grouped by {split_col} saved to {target_file}")
    except Exception as e:
        logger.error(f"Error while creating data grouped by {split_col}: {e}")
        raise RuntimeError(f"Failed to create data: {e}")
