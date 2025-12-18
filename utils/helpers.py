"""
Helper functions for text processing and visualization
"""

import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here'}

def tokenize_text(text: str) -> list:
    """
    Tokenize text (equivalent to PySpark Tokenizer)
    
    Parameters:
    text (str): Input text to tokenize
    Returns:
    list: List of tokens
    """
    
    if pd.isna(text):
        return []    
    
    # Convert to lowercase and split by whitespace
    text = str(text).lower()
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def preprocess_text(text) -> str:
    """
    Complete text preprocessing pipeline

    Parameters:
    text (str): Input text to preprocess
    
    Returns:
    str: Preprocessed text
    """

    tokens = tokenize_text(text)
    if not tokens:
        return ''
    
    filtered_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return ' '.join(filtered_tokens)

def plot_hist(file_name, partisanship) -> None:
    """
    Plot histogram of probability distribution.

    Parameters:
    file_name (str): Path to the CSV file containing the data.
    partisanship (str): 'republican' or 'democrat' to specify which histogram to plot.
    """

    is_republican = partisanship.strip().lower() == 'republican'

    img_name = "rep_prob_hist.png" if is_republican else "dem_prob_hist.png"
    y_col = 'prob_republican' if is_republican else 'prob_democrat'
    y_label = 'P(Republican)' if is_republican else 'P(Democrat)'
    title = f"Histogram of {partisanship.title()} Probabilities"
    
    print(f"Plotting {title}")
    df = pd.read_csv(file_name)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, x=y_col, hue='label', 
        bins=25, kde=False, stat="count", common_norm=False
    )
    plt.title(title)
    plt.xlabel(y_label)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()
