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

def train_model(split_col: str):
    """
    Train the classification model on the training data
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    train_data, train_model = ISSUE_TRAIN, ISSUE_TRAINED_MODEL
    if split_col == 'series':
        train_data, train_model = SERIES_TRAIN, SERIES_TRAINED_MODEL
    if split_col == 'para':
        train_data, train_model = PARA_TRAIN, PARA_TRAINED_MODEL
    
    chunks = pd.read_csv(train_data, chunksize=CHUNK_SIZE)
    train_baseline(chunks=chunks, model_loc=train_model, stopwords=STOPWORDS)

def eval_model(split_col: str):
    """
    Evaluate the trained model on the test data
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    test_data, train_model = ISSUE_TEST, ISSUE_TRAINED_MODEL
    if split_col == 'series':
        test_data, train_model = SERIES_TEST, SERIES_TRAINED_MODEL
    if split_col == 'para':
        test_data, train_model = PARA_TEST, PARA_TRAINED_MODEL

    test_chunks = pd.read_csv(test_data, chunksize=CHUNK_SIZE)
    eval_performance(model_loc=train_model, chunks=test_chunks, stopwords=STOPWORDS)

def get_logits(split_col: str):
    """
    Get logits for the test data using the trained model
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """
    
    test_file, model_loc, logits_file = ISSUE_TEST, ISSUE_TRAINED_MODEL, ISSUE_LOGITS_DATA
    if split_col == 'series':
        test_file, model_loc, logits_file = SERIES_TEST, SERIES_TRAINED_MODEL, SERIES_LOGITS_DATA
    if split_col == 'para':
        test_file, model_loc, logits_file = PARA_TEST, PARA_TRAINED_MODEL, PARA_LOGITS_DATA
    
    test_chunks = pd.read_csv(test_file, chunksize=CHUNK_SIZE)
    logits_chunks = create_logits(chunks=test_chunks, model_loc=model_loc, stopwords=STOPWORDS)
    if logits_chunks:
        logits_df = pd.concat(logits_chunks, ignore_index=True)
        logits_df.to_csv(logits_file, index=False)
        logger.info(f"Logits saved to {logits_file}")

def create_prob_df(split_col: str):
    """
    Create probability DataFrame from logits
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    source_file, target_file = ISSUE_MAX_POOL_DATA, ISSUE_PROB_DATA
    if split_col == 'series':
        source_file = SERIES_MAX_POOL_DATA
        target_file = SERIES_PROB_DATA
    if split_col == 'para':
        source_file = PARA_MAX_POOL_DATA
        target_file = PARA_PROB_DATA

    df = pd.read_csv(source_file)
    df['prob_republican'] = 1 / (1 + np.exp(-df['logits']))
    df['prob_democrat'] = 1 - df['prob_republican']
    df.to_csv(target_file)

def create_max_pool_df(split_col: str) -> None:
    """
    Perform max pooling based on absolute logit values
    
    Parameters:
    split_col (str): Column name to split on for unique values.
    """

    final_col_names = ["series", "issue", "new_text", "processed_text", "logits", "label", "id", "date"]
    source_file, target_file = ISSUE_LOGITS_DATA, ISSUE_MAX_POOL_DATA
    if split_col == 'series':
        source_file = SERIES_LOGITS_DATA
        target_file = SERIES_MAX_POOL_DATA
    if split_col == 'para':
        source_file = PARA_LOGITS_DATA
        target_file = PARA_MAX_POOL_DATA

    # read source file
    logits_df = pd.read_csv(source_file)
    # calculate absolute logit
    logits_df['abs_logit'] = logits_df['logits'].abs()
    # sort by issue and absolute logit descending
    df_sorted = logits_df.sort_values(by=['issue', 'abs_logit'], ascending=[True, False])
    # drop duplicates, keeping most confident prediction for each issue
    issue_df = df_sorted.drop_duplicates(subset=['issue'], keep='first')[final_col_names]
    # assign predicted label based on logit sign
    issue_df['predicted_label'] = (issue_df['logits'] > 0).astype(int)
    # save to csv
    issue_df.to_csv(target_file, index=False)
    print(f"Max pooled data saved to {target_file}")

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
    train_model(split_col='issue')
    # evaluate model
    eval_model(split_col='issue')
    # get logits for test data
    get_logits(split_col='issue')
    # perform max pooling
    create_max_pool_df(split_col='issue')
    # create probabilities
    create_prob_df(split_col='issue')
    # visualise probability distribution
    plot_hist(file_name=ISSUE_PROB_DATA, partisanship='republican')
