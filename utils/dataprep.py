"""
Main code file to prepare training data
"""

import glob
import pandas as pd

from pyspark.sql import functions as F

def create_data(spark_session: any, parquet_loc: str) -> pd.DataFrame:
    """
    Create initial dataframe from parquet file by splitting into paragraphs and filtering by date
    
    Parameters:
    parquet_loc (str): Location of the parquet file
    spark_session (any): Spark session object
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
        raise RuntimeError(f"An error occurred while creating data from {parquet_loc}: {e}")

def process_metadata(spark_session: any, csv_file: str) -> pd.DataFrame:
    """
    Process metadata CSV to assign labels and filter out unwanted labels
    
    Parameters:
    csv_file (str): Location of the metadata CSV file
    spark_session (any): Spark session object
    Returns:
    csv_df: Spark DataFrame with columns series, contents, label
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
        csv_df = csv_df.select('series', F.col('contents'), F.col('label'))
        return csv_df
    except Exception as e:
        raise RuntimeError(f"Failed to process metadata {csv_file}: {e}")

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
        return None
    
    try:
        batches, temp_batch = [], []
        # read the first file with header to get column names
        first_df = pd.read_csv(csv_files[0])

        for idx, file in enumerate(csv_files[1:], 1):
            # read the rest without header
            temp_df = pd.read_csv(file, header=None, names=first_df.columns)
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
        
        all_dfs = [[first_df][0]] + batches
        result = pd.concat(all_dfs, ignore_index=True)
        return result
    except Exception as e:
        raise RuntimeError(f"An error occurred while concatenating all data: {e}")
