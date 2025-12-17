"""

"""

import os
import logging
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from pyspark.sql import SparkSession

from utils.consts import *
from utils.helpers import preprocess_text, STOPWORDS, create_max_pool_df, plot_hist
from utils.dataprep import *
from utils.data_splitting import split_data
from model import train_baseline, eval_performance, create_logits

def setup_env():
    """Setup environment variables from .env file."""
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    dirs = [
        os.getenv('MASTER_DIR'),
        os.getenv('MODEL_STORAGE'),
        os.path.dirname(PREPARED_DATA)
    ]

    for dir in filter(None, dirs):
        os.makedirs(dir, exist_ok=True)
    
    return logging.getLogger(__name__)

def init_data(parquet_path: str, metadata_path: str, output_dir: str):
    """Initialize data by reading parquet and metadata, joining, and saving to CSV."""

    # initialize spark session
    spark = SparkSession.builder.appName("PartisianClassification").getOrCreate()

    # read data
    df = create_data(spark_session=spark, parquet_loc=parquet_path)
    if df is None:
        print("Data initialization failed, could not read Parquet")
        spark.stop()
        return
    print(f"Read corpus from: {parquet_path}")
    
    # read metadata
    csv_df = process_metadata(spark_session=spark, csv_file=metadata_path)
    if csv_df is None:
        print("Data initialization failed, could not read Metadata CSV")
        spark.stop()
        return
    print(f"Read metadata from: {metadata_path}")

    # inner join
    df = df.join(csv_df, on='series', how='inner')
    # save multiple csv files in directory
    df.write.csv(output_dir, header=True, escape='"')
    print(f"Joined data saved to directory: {output_dir}")
    
    # concatenate into single dataframe
    concat_df = concatenate(data_dir=output_dir)
    if concat_df is None:
        print("Data concatenation failed")
        spark.stop()
        return
    print("Data concatenation successful")
    
    # save to csv
    concat_df.to_csv(PREPARED_DATA, index=False, header=True, quoting=1)
    print(f"Prepared data saved to: {PREPARED_DATA}")
    
    spark.stop()

def preprocess():
    """Pre-process the prepared training data and save to CSV"""
    
    cols_to_keep = ['series', 'issue', 'date', 'id', 'new_text', 'processed_text', 'label']
    
    try:
        # read master data
        training_df = pd.read_csv(PREPARED_DATA)
        # Convert date column if it exists
        if 'date' in training_df.columns:
            training_df['date'] = pd.to_datetime(training_df['date'], errors='coerce')
        # Fill missing values
        training_df['new_text'] = training_df['new_text'].fillna('')
        # Preprocess text
        training_df['processed_text'] = training_df['new_text'].apply(preprocess_text)
        # keep only relevant columns and drop NA
        training_df = training_df[cols_to_keep].dropna()
        # filter out texts with fewer than 100 words
        training_df = training_df[training_df['processed_text'].apply(lambda x: len(str(x).split()) >= 100)]
        # save to csv
        training_df.to_csv(PROCESSED_DATA, index=False)
        logger.info(f"Pre-processed data saved to: {PROCESSED_DATA}")
    except Exception as e:
        logger.error(f"An error occurred during pre processing: {e}")

def train_model():
    """Train the classification model on the training data"""
    
    chunks = pd.read_csv(ISSUE_TRAIN, chunksize=CHUNK_SIZE)
    train_baseline(chunks=chunks, model_loc=ISSUE_TRAINED_MODEL, stopwords=STOPWORDS)
    # chunks = pd.read_csv(SERIES_TRAIN, chunksize=CHUNK_SIZE)
    # train_baseline(chunks=chunks, model_loc=SERIES_TRAINED_MODEL)
    # chunks = pd.read_csv(PARA_TRAIN, chunksize=CHUNK_SIZE)
    # train_baseline(chunks=chunks, model_loc=PARA_TRAINED_MODEL)

def eval_model():
    """Evaluate the trained model on the test data"""
    test_chunks = pd.read_csv(ISSUE_TEST, chunksize=CHUNK_SIZE)
    eval_performance(model_loc=ISSUE_TRAINED_MODEL, chunks=test_chunks, stopwords=STOPWORDS)    
    # test_chunks = pd.read_csv(SERIES_TEST, chunksize=CHUNK_SIZE)
    # eval_performance(model_loc=SERIES_TRAINED_MODEL, chunks=test_chunks, stopwords=STOPWORDS)
    # test_chunks = pd.read_csv(PARA_TEST, chunksize=CHUNK_SIZE)
    # eval_performance(model_loc=PARA_TRAINED_MODEL, chunks=test_chunks, stopwords=STOPWORDS)

def get_logits():
    """Get logits for the test data using the trained model"""
    test_chunks = pd.read_csv(ISSUE_TEST, chunksize=CHUNK_SIZE)
    logits_chunks = create_logits(chunks=test_chunks, model_loc=ISSUE_TRAINED_MODEL, stopwords=STOPWORDS)
    # test_chunks = pd.read_csv(SERIES_TEST, chunksize=CHUNK_SIZE)
    # logits_chunks = create_logits(chunks=test_chunks, model_loc=SERIES_TRAINED_MODEL, stopwords=STOPWORDS)
    # test_chunks = pd.read_csv(PARA_TEST, chunksize=CHUNK_SIZE)
    # logits_chunks = create_logits(chunks=test_chunks, model_loc=PARA_TRAINED_MODEL, stopwords=STOPWORDS)
    
    if logits_chunks:
        logits_df = pd.concat(logits_chunks, ignore_index=True)
        logits_df.to_csv(ISSUE_LOGITS_DATA, index=False)
        print("Logits saved to", ISSUE_LOGITS_DATA)
        # logits_df.to_csv(SERIES_LOGITS_DATA, index=False)
        # print("Logits saved to", SERIES_LOGITS_DATA)
        # logits_df.to_csv(PARA_LOGITS_DATA, index=False)
        # print("Logits saved to", PARA_LOGITS_DATA)

def create_prob_df(source_file: str, target_file: str):
    """
    Create probability DataFrame from logits
    
    Parameters:
    source_file (str): Path to the source CSV file with logits
    target_file (str): Path to the target CSV file to save probabilities
    """

    df = pd.read_csv(source_file)
    df['prob_republican'] = 1 / (1 + np.exp(-df['logits']))
    df['prob_democrat'] = 1 - df['prob_republican']
    df.to_csv(target_file)

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
    
    # split into train and test data
    split_data(file_name=PROCESSED_DATA, train_data_loc=ISSUE_TRAIN, test_data_loc=ISSUE_TEST, split_col='issue')
    # split_data(file_name=PROCESSED_DATA, train_data_loc=SERIES_TRAIN, test_data_loc=SERIES_TEST, split_col='series')
    # split_data(file_name=PROCESSED_DATA, train_data_loc=PARA_TRAIN, test_data_loc=PARA_TEST, stratify=True)
    
    # train model
    train_model()
    
    # evaluate model
    eval_model()
    
    # get logits for test data
    get_logits()
    
    # perform max pooling
    create_max_pool_df(source_file=ISSUE_LOGITS_DATA, target_file=ISSUE_MAX_POOL_DATA)
    # create_max_pool_df(source_file=SERIES_LOGITS_DATA, target_file=SERIES_MAX_POOL_DATA)
    # create_max_pool_df(source_file=PARA_LOGITS_DATA, target_file=PARA_MAX_POOL_DATA)
    
    # create probabilities
    create_prob_df(source_file=ISSUE_MAX_POOL_DATA, target_file=ISSUE_PROB_DATA)
    # create_prob_df(source_file=SERIES_MAX_POOL_DATA, target_file=SERIES_PROB_DATA)
    # create_prob_df(source_file=PARA_MAX_POOL_DATA, target_file=PARA_PROB_DATA)

    # visualise probability distribution
    plot_hist(file_name=ISSUE_PROB_DATA, partisanship='republican')
