"""
Split data into training and testing sets based on unique issues, series, 
or random paragraph split.
"""

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def _save_split_data(
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        train_path: str,
        test_path: str,
        description: str = ""
    ) -> None:
    """
    Save train and test DataFrames to CSV files.

    Parameters:
    train_df (pd.DataFrame): Training data DataFrame.
    test_df (pd.DataFrame): Testing data DataFrame.
    train_path (str): Path to save the training data CSV.
    test_path (str): Path to save the testing data CSV.
    description (str, optional): Description for logging purposes.
    """

    prefix = f"{description}" if description else ""

    train_df.to_csv(train_path, index=False)
    logger.info(f"{prefix} Training data saved to: {train_path}")
    test_df.to_csv(test_path, index=False)
    logger.info(f"{prefix} Testing data saved to: {test_path}")

def split_data(
        file_name: str, 
        train_data_loc: str, 
        test_data_loc: str, 
        split_col : Optional[str] = None, 
        stratify: bool =False
    ) -> None:
    """
    Generic function to split data into training and testing sets.
    
    Parameters:
    file_name (str): Path to the CSV file containing the data.
    train_data_loc (str): Path to save the training data CSV.
    test_data_loc (str): Path to save the testing data CSV.
    split_col (str, optional): Column name to split on for unique values. If None, random split is performed.
    stratify (bool, optional): Whether to stratify the split based on the 'label' column. Default is False.
    """

    try:
        df = pd.read_csv(file_name)
        if split_col:
            # Get unique ids
            unique_ids = df[split_col].unique()
            # Split issue into train and test
            train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=0)
            # Filter dataframes based on issue split
            train_df = df[df[split_col].isin(train_ids)].copy()
            test_df = df[df[split_col].isin(test_ids)].copy()
        else:
            stratify_col = df['label'] if stratify else None
            # Split dataframe into train and test
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=stratify_col)
        
        # Ensure labels are numeric
        train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
        test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce')

        _save_split_data(
            train_df=train_df, 
            test_df=test_df, 
            train_path=train_data_loc, 
            test_path=test_data_loc, 
        )
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

def split_doc_data(
        doc_data: str, 
        split_col: str, 
        train_data_path: str, 
        test_data_path: str
    ) -> None:
    """
    Split the document-level data into training and testing sets based on unique issues or series.

    Parameters:
    doc_data (str): Path to the CSV file containing document-level data.
    split_col (str): Column name to split on for unique values ('issue' or 'series').
    train_data_path (str): Path to save the training data CSV.
    test_data_path (str): Path to save the testing data CSV.
    """

    try:
        df = pd.read_csv(doc_data)
        sep_label = df.groupby(split_col)['label'].first()
        unique_labels = sep_label.index.values
        train_split, test_split = train_test_split(
            unique_labels, 
            test_size=0.2, 
            random_state=0, 
            stratify=sep_label.values
        )
        train_df = df[df[split_col].isin(train_split)].reset_index(drop=True).copy()
        test_df = df[df[split_col].isin(test_split)].reset_index(drop=True).copy()
        
        _save_split_data(
            train_df=train_df, 
            test_df=test_df, 
            train_path=train_data_path, 
            test_path=test_data_path, 
            description="Document-level"
        )
    except Exception as e:
        logger.error(f"Error during document data splitting: {e}")
        raise
