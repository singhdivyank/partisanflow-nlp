"""
Main entry point for Partisan Classification pipeline.
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.consts import CHUNK_SIZE, PREPARED_DATA, PROCESSED_DATA
from utils.data_splitting import split_data
from utils.dataprep import concatenate, create_data, process_metadata
from utils.helpers import STOPWORDS, plot_hist, preprocess_text
from utils.logger_config import setup_logger
from utils.model import create_logits, eval_performance, train_baseline
from utils.path_utils import get_para_paths
from utils.spark_session import get_spark_session, stop_spark_session

logger = setup_logger(__name__)

def setup_env() -> None:
    """Setup environment variables from .env file."""
    load_dotenv()

    dirs = [
        os.getenv('MASTER_DIR'),
        os.getenv('MODEL_STORAGE'),
        os.path.dirname(PREPARED_DATA)
    ]

    for d in filter(None, dirs):
        os.makedirs(d, exist_ok=True)

def init_data(parquet_path: str, metadata_path: str, output_dir: str) -> None:
    """
    Initialize data by reading parquet and metadata, joining, and saving to CSV.
    
    Parameters:
    parquet_path (str): Path to the Parquet file.
    metadata_path (str): Path to the metadata CSV file.
    output_dir (str): Directory to save the intermediate CSV files.
    """

    # initialize spark session
    spark = get_spark_session()

    try:
        # read data
        df = create_data(spark_session=spark, parquet_loc=parquet_path)
        logger.info(f"Read corpus from: {parquet_path}")
        
        # read metadata
        csv_df = process_metadata(spark_session=spark, csv_file=metadata_path)
        logger.info(f"Read metadata from: {metadata_path}")

        # inner join
        df = df.join(csv_df, on='series', how='inner')
        # save multiple csv files in directory
        df.write.csv(output_dir, header=True, escape='"')
        logger.info(f"Joined data saved to directory: {output_dir}")
        
        # concatenate into single dataframe
        concat_df = concatenate(data_dir=output_dir)
        if concat_df is None:
            logger.error("Data concatenation failed")
            return
        
        # save to csv
        concat_df.to_csv(PREPARED_DATA, index=False, header=True, quoting=1)
        logger.info(f"Prepared data saved to: {PREPARED_DATA}")
    except Exception as e:
        logger.error(f"Data initialization failed: {e}")
    finally:
        stop_spark_session()

def preprocess() -> None:
    """Pre-process the prepared training data and save to CSV"""
    
    cols_to_keep = [
        'series', 'issue', 'date', 'id', 
        'new_text', 'processed_text', 'label'
    ]
    
    try:
        # read master data
        training_df = pd.read_csv(PREPARED_DATA)
        # Convert date column if it exists
        if 'date' in training_df.columns:
            training_df['date'] = pd.to_datetime(
                training_df['date'], errors='coerce'
            )
        # Fill missing values
        training_df['new_text'] = training_df['new_text'].fillna('')
        # Preprocess text
        training_df['processed_text'] = training_df['new_text'].apply(preprocess_text)
        # keep only relevant columns and drop NA
        training_df = training_df[cols_to_keep].dropna()
        # filter out texts with fewer than 100 words
        training_df = training_df[
            training_df['processed_text'].apply(lambda x: len(str(x).split()) >= 100)
        ]
        # save to csv
        training_df.to_csv(PROCESSED_DATA, index=False)
        logger.info(f"Pre-processed data saved to: {PROCESSED_DATA}")
    except Exception as e:
        logger.error(f"An error occurred during pre processing: {e}")

def train_model(split_col: str) -> None:
    """
    Train the classification model on the training data
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    paths = get_para_paths(split_col)
    chunks = pd.read_csv(paths['train'], chunksize=CHUNK_SIZE)
    train_baseline(chunks=chunks, model_loc=paths['model'], stopwords=STOPWORDS)

def eval_model(split_col: str) -> None:
    """
    Evaluate the trained model on the test data
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    paths = get_para_paths(split_col)
    test_chunks = pd.read_csv(paths['test'], chunksize=CHUNK_SIZE)
    eval_performance(model_loc=paths['model'], chunks=test_chunks, stopwords=STOPWORDS)

def get_logits(split_col: str) -> None:
    """
    Get logits for the test data using the trained model
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """
    
    paths = get_para_paths(split_col)
    test_chunks = pd.read_csv(paths['test'], chunksize=CHUNK_SIZE)
    logits_chunks = create_logits(chunks=test_chunks, model_loc=paths['model'], stopwords=STOPWORDS)
    if logits_chunks:
        logits_df = pd.concat(logits_chunks, ignore_index=True)
        logits_df.to_csv(paths['logits'], index=False)
        logger.info(f"Logits saved to {paths['logits']}")

def create_prob_df(split_col: str) -> None:
    """
    Create probability DataFrame from logits
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    paths = get_para_paths(split_col)
    prob_df = pd.read_csv(paths['max_pool'])
    prob_df['prob_republican'] = 1 / (1 + np.exp(-prob_df['logits']))
    prob_df['prob_democrat'] = 1 - prob_df['prob_republican']
    prob_df.to_csv(paths['prob'], index=False)
    logger.info(f"Predicted probability scores saved to {paths['prob']}")

def create_max_pool_df(split_col: str) -> None:
    """
    Perform max pooling based on absolute logit values
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    final_col_names = [
        "series", "issue", "new_text", "processed_text", 
        "logits", "label", "id", "date"
    ]

    paths = get_para_paths(split_col)
    # read source file
    logits_df = pd.read_csv(paths['logits'])
    # calculate absolute logit
    logits_df['abs_logit'] = logits_df['logits'].abs()
    # sort by issue and absolute logit descending
    df_sorted = logits_df.sort_values(by=['issue', 'abs_logit'], ascending=[True, False])
    # drop duplicates, keeping most confident prediction for each issue
    issue_df = df_sorted.drop_duplicates(subset=['issue'], keep='first')[final_col_names]
    # assign predicted label based on logit sign
    issue_df['predicted_label'] = (issue_df['logits'] > 0).astype(int)
    # save to csv
    issue_df.to_csv(paths['max_pool'], index=False)
    logger.info(f"Max pooled data saved to {paths['max_pool']}")

def run_pipeline(split_col: str, stratify: bool = False) -> None:
    """
    Runs the complete classification pipeline for a given split-type.

    Parameters:
    split_col (str): Column name to split on for unique values ('issue' or 'series').
    stratify (bool, optional): Whether to stratify the split based on the 'label' column. Default is False.
    """

    paths = get_para_paths(split_col=split_col)
    logger.info(f"Running pipeline for {split_col}")

    # split into train and test data
    split_kwargs = {'split_col': split_col} if split_col != 'para' else {'stratify': stratify}
    split_data(
        file_name=PROCESSED_DATA, train_data_loc=paths['train'], 
        test_data_loc=paths['test'], **split_kwargs
    )
    # train model
    train_model(split_col=split_col)
    # evaluate model
    eval_model(split_col=split_col)
    # get logits for test data
    get_logits(plit_col=split_col)
    # perform max pooling
    create_max_pool_df(plit_col=split_col)
    # create probabilities
    create_prob_df(plit_col=split_col)
    # visualise probability distribution
    plot_hist(file_name=paths['prob'], partisanship='republican')

    logger.info(f"Pipeline for {split_col} completed.")

if __name__=='__main__':

    logger = setup_env()
    # initialize data
    init_data(
        parquet_path = os.getenv('PARQUET_FILE'),
        metadata_path = os.getenv('METADATA_CSV'),
        output_dir = os.getenv('MASTER_DIR')
    )
    # pre-process data
    preprocess()
    # run paragraph-level pipelines
    run_pipeline(split_col='issue')
    run_pipeline(split_col='series')
    run_pipeline(split_col='para', stratify=True)
