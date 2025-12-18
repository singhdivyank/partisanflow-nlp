"""
Main entry point for document partisanship pipeline.
"""

import os

import numpy as np
import pandas as pd

from utils.consts import CHUNK_SIZE, DOC_PATH, PROCESSED_DATA
from utils.data_splitting import split_doc_data
from utils.dataprep import create_doc_data
from utils.helpers import STOPWORDS
from utils.logger_config import setup_logger
from utils.model import train_doc_data_clf, create_logits, eval_performance
from utils.path_utils import get_doc_paths

logger = setup_logger(__name__)

def ensure_dictionaries() -> None:
    """Create necessary directories if they do not exist."""
    os.makedirs(DOC_PATH, exist_ok=True)

def compute_and_save_logits(test_path: str, model_path: str, logits_path: str) -> None:
    """
    Compute logits for test data and save to CSV.
    
    Parameters:
    test_path (str): Path to the test data CSV file.
    model_path (str): Path to the trained model pickle file.
    logits_path (str): Path to save the logits CSV file.
    """
    chunks = pd.read_csv(test_path, chunksize=CHUNK_SIZE)
    logits_chunks = create_logits(
        chunks=chunks, 
        model_loc=model_path, 
        stopwords=STOPWORDS
    )
    if not logits_chunks:
        logger.warning("No logits generated")
    else:
        logits_df = pd.concat(logits_chunks, ignore_index=True)
        logits_df.to_csv(logits_path, index=False)
        logger.info(f"Logits saved to {logits_path}")

def create_prob_df(source_file: str, target_file: str) -> None:
    """
    Create probability DataFrame from logits.
    
    Parameters:
    source_file (str): Path to the CSV file containing logits.
    target_file (str): Path to save the probability scores CSV file.
    """

    df = pd.read_csv(source_file)
    df['prob_republican'] = 1 / (1 + np.exp(-df['logits']))
    df['prob_democrat'] = 1 - df['prob_republican']
    df.to_csv(target_file, index=False)
    logger.info(f"Predicted probability scores saved to {target_file}")

def run_doc_pipeline(split_col: str) -> None:
    """
    Run the complete document-level pipeline for a given split column.

    Parameters:
    split_col (str): Column name to split on for unique values ('issue' or 'series').
    """

    paths = get_doc_paths(split_col=split_col)

    logger.info(f"Running document pipeline for {split_col}")

    create_doc_data(
        split_col=split_col, source_file=PROCESSED_DATA, 
        target_file=paths['doc_data']
    )
    split_doc_data(
        doc_data=paths['doc_data'], 
        split_col=split_col, 
        train_data_path=paths['train'], 
        test_data_path=paths['test']
    )
    train_doc_data_clf(
        doc_data_path=paths['train'], 
        model_loc=paths['model'], 
        stopwords=STOPWORDS
    )
    eval_chunks = pd.read_csv(paths['test'], chunksize=CHUNK_SIZE)
    eval_performance(
        model_loc=paths['model'], 
        chunks=eval_chunks, 
        stopwords=STOPWORDS
    )
    compute_and_save_logits(
        test_path=paths['test'], 
        model_path=paths['model'], 
        logits_path=paths['logits']
    )
    create_prob_df(
        source_file=paths['logits'],
        target_file=paths['prob']
    )

    logger.info(f"Document pipeline for {split_col} completed.")
    


if __name__ == '__main__':
    
    ensure_dictionaries()
    run_doc_pipeline(split_col='issue')
    run_doc_pipeline(split_col='series')
