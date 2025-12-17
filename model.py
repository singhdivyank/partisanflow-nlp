"""
All functions related to model training, making predictions, and evaluation.
"""

import numpy as np
import pickle

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def get_vectorizer(stopwords: set):
    """Vectorizer initialization function"""
    return HashingVectorizer(n_features=10000, stop_words=stopwords, alternate_sign=False, norm='l2')

def train_baseline(chunks: list, model_loc: str, stopwords: set):
    """
    Train baseline classifier on chunks and save as a pickle file
    
    Parameters:
    chunks (list): list of DataFrame chunks containing 'processed_text' and 'label' columns.
    model_loc (str): Path to save the trained model pickle file.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    """

    # initialise classifier
    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=0, alpha=1e-4)
    # initialise label encoder
    encoder = LabelEncoder()
    # initialize HashingVectorizer
    cv = HashingVectorizer(n_features=10000, stop_words=stopwords, alternate_sign=False, norm='l2')
    first_chunk = True
    
    try:
        for idx, chunk in enumerate(chunks):
            # Transform using the HashingVectorizer
            X = cv.transform(chunk["processed_text"].astype(str))
            # Encode and prepare labels
            y = encoder.fit_transform(chunk["label"])
            classes = np.array([0, 1]) if first_chunk else None
            # partial_fit
            clf.partial_fit(X, y, classes=classes)
            first_chunk = False
            logger.info(f"Trained on chunk {idx+1}. Total rows: {len(chunk)}")
        
        # save the trained model
        with open(model_loc, 'wb') as f:
            pickle.dump({'encoder': encoder, 'cv': cv, 'clf': clf}, f)
    
        logger.info(f"Model saved to: {model_loc}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")

def load_model(model_loc: str, component_key: str):
    """Helper function to load specific parts of saved model."""
    try:
        with open(model_loc, 'rb') as f:
            return pickle.load(f).get(component_key, None)
    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_loc}")
        return None

def make_preds(chunks: list, clf, cv) -> None:
    """
    Make predictions on the test data
    
    Parameters:
    chunks (list): list of DataFrame chunks containing 'processed_text' and 'label' columns.
    clf: Trained classifier
    cv: Trained HashingVectorizer
    """
    
    y_true, y_pred = [], []

    try:
        # Iterate over the large CSV file in chunks for testing
        for i, chunk in enumerate(chunks):
            X_test = cv.transform(chunk["processed_text"].astype(str))
            # model outputs
            y_pred_chunk = clf.predict(X_test)
            y_true.extend(chunk["label"].tolist())
            y_pred.extend(y_pred_chunk.tolist())
            logger.info(f"Tested on chunk {i+1}. Total test rows processed: {len(chunk)}")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
    
    return y_true, y_pred

def eval_performance(model_loc: str, chunks: list, stopwords: set) -> None:
    """
    Evaluate model performance on test data
    
    Parameters:
    model_loc (str): Path to the trained model pickle file.
    chunks (list): list of DataFrame chunks containing 'processed_text' and 'label' columns.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    """

    accuracy, report = 0.0, None

    # load the trained model
    clf = load_model(model_loc, 'clf')
    if clf is None:
        logger.info(f"Accuracy: {accuracy}\nClassifier Report: {report}")
        return
    
    # initialize HashingVectorizer
    cv = get_vectorizer(stop_words=stopwords)
    y_true, y_pred = make_preds(chunks=chunks, clf=clf, cv=cv)
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['0', '1'], zero_division=0)
    
    logger.info(f"Accuracy: {accuracy:.4f}\nClassifier Report: {report}")

def create_logits(chunks: list, model_loc: str, stopwords: set) -> list:
    """
    Get logits for the entire test dataset
    
    Parameters:
    model_loc (str): Path to the trained model pickle file.
    chunks (list): list of DataFrame chunks containing 'processed_text' column.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    Returns:
    list: List of DataFrame chunks with added 'logits' column.
    """
    
    logit_chunks = []
    # initialize HashingVectorizer
    cv = get_vectorizer(stop_words=stopwords)
    # load the trained model
    clf = load_model(model_loc, 'clf')
    if clf is None:
        return logit_chunks

    try:
        # Iterate over the large CSV file in chunks for testing
        for idx, chunk in enumerate(chunks):
            X_test = cv.transform(chunk["processed_text"].astype(str))
            # model outputs
            logit = clf.decision_function(X_test)
            chunk['logits'] = logit
            logit_chunks.append(chunk)
            logger.info(f"Processed chunk {idx+1}. Rows processed: {len(chunk)}. Logit: {logit}")
    except Exception as e:
        logger.error(f"Error in logits extraction: {e}")
    
    return logit_chunks
