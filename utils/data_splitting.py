"""
Split data into training and testing sets based on unique issues, series, or random paragraph split.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

def split_data(file_name: str, train_data_loc: str, test_data_loc: str, split_col=None, stratify=False):
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

        # save to csv
        train_df.to_csv(train_data_loc, index=False)
        logger.info(f"Training data saved to: {train_data_loc}")
        test_df.to_csv(test_data_loc, index=False)
        logger.info(f"Testing data saved to: {test_data_loc}")
    except Exception as e:
        logger.error(f"An error occurred during data splitting: {e}")
