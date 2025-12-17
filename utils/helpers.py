"""
Class with helper functions
"""

import matplotlib.pyplot as plt
import pandas as pd
import re
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

def plot_hist(file_name, partisanship):
    img_name = "rep_prob_hist.png" if partisanship == 'republican' else "dem_prob_hist.png"
    y_col = 'prob_republican' if partisanship == 'republican' else 'prob_democrat'
    y_label = 'P(Republican)' if partisanship == 'republican' else 'P(Democrat)'
    title = "Histogram of Republican Probabilities" if partisanship == 'republican' else "Histogram of Democrat Probabilities"
    
    print(f"Plotting {partisanship.title()} Histogram")
    df = pd.read_csv(file_name)

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=y_col, hue='label', bins=25, kde=False, stat="count", common_norm=False)
    plt.title(title)
    plt.xlabel(y_label)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()
