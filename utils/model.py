"""
Model training, prediction, and evaluation functions.
"""

import pickle
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def get_vectorizer(stopwords: set) -> HashingVectorizer:
    """
    Vectorizer initialization function
    
    Parameters:
    stopwords (set): Set of stopwords to exclude.

    Returns:
    HashingVectorizer: Configured HashingVectorizer instance.
    """
    return HashingVectorizer(
        n_features=10000, 
        stop_words=stopwords, 
        alternate_sign=False, 
        norm='l2'
    )

def save_model(model_loc: str, model_data: Any) -> None:
    """
    Helper function to save model components as a pickle file.
    
    Parameters:
    model_loc (str): Path to save the trained model pickle file.
    model_data (Any): Model data to be saved (could be dict of components).
    """

    try:
        with open(model_loc, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to: {model_loc}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model(
        model_loc: str, 
        component_key: Optional[str] = None
    ) -> Optional[Any]:
    """
    Helper function to load specific parts of saved model.
    
    Parameters:
    model_loc (str): Path to the trained model pickle file.
    component_key (Optional[str]): Key of the component to retrieve ('encoder', 'cv', 'clf').

    Returns:
    Optional[Any]: The requested component of the model, or None if not found.
    """

    try:
        with open(model_loc, 'rb') as f:
            model_data = pickle.load(f)
            
        if component_key is None:
            return model_data
        
        if isinstance(model_data, dict):
            return model_data.get(component_key)

        return model_data
    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_loc}")
        return None

def train_baseline(
        chunks: Iterator[pd.DataFrame],
        model_loc: str, 
        stopwords: set
    ) -> None:
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
    cv = get_vectorizer(stopwords=stopwords)
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
        save_model(model_loc, {'encoder': encoder, 'cv': cv, 'clf': clf})
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def train_doc_data_clf(
        doc_data_path: str, 
        model_loc: str, 
        stopwords: set
    ) -> None:
    """
    Train classifier on document-level data and save as a pickle file.

    Parameters:
    doc_data_path (str): Path to the CSV file containing document-level data.
    model_loc (str): Path to save the trained model pickle file.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    """

    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=0, alpha=1e-4)
    cv = get_vectorizer(stopwords=stopwords)

    try:
        df = pd.read_csv(doc_data_path)
        X = cv.transform(df["processed_text"])
        y = df["label"].values
        clf.fit(X, y)
        logger.info("Training complete on issue-level data.")
        save_model(model_loc, {'cv': cv, 'clf': clf})
    except Exception as e:
        logger.error(f"Document-level model training failed: {e}")
        raise

def make_preds(
        chunks: Iterator[pd.DataFrame], 
        clf: SGDClassifier, 
        cv: HashingVectorizer
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on the test data
    
    Parameters:
    chunks (Iterator): list of DataFrame chunks containing 'processed_text' and 'label' columns.
    clf (SGDClassifier): Trained classifier
    cv (HashingVectorizer): Trained HashingVectorizer
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
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
    
    return np.array(y_true), np.array(y_pred)

def eval_performance(
        model_loc: str, 
        chunks: Iterator[pd.DataFrame], 
        stopwords: set
    ) -> None:
    """
    Evaluate model performance on test data
    
    Parameters:
    model_loc (str): Path to the trained model pickle file.
    chunks (Iterator): list of DataFrame chunks containing 'processed_text' and 'label' columns.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    """

    # load the trained model
    clf = load_model(model_loc, 'clf')
    if clf is None:
        logger.info(f"Accuracy: 0.0\nClassifier Report: None")
        return
    
    # initialize HashingVectorizer
    cv = get_vectorizer(stopwords=stopwords)
    y_true, y_pred = make_preds(chunks=chunks, clf=clf, cv=cv)
    if len(y_true) > 0 and len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=['0', '1'], zero_division=0
        )
        logger.info(f"Accuracy: {accuracy:.4f}\nClassifier Report: {report}")
    else:
        logger.info("Accuracy: 0.0\nClassifier Report: None")

def create_logits(
        chunks: Iterator[pd.DataFrame], 
        model_loc: str, 
        stopwords: set
    ) -> List[pd.DataFrame]:
    """
    Get logits for the entire test dataset
    
    Parameters:
    chunks (Iterator): list of DataFrame chunks containing 'processed_text' column.
    model_loc (str): Path to the trained model pickle file.
    stopwords (set): Set of stopwords to be used in HashingVectorizer.
    
    Returns:
    List: List of DataFrame chunks with added 'logits' column.
    """
    
    logit_chunks = []
    # initialize HashingVectorizer
    cv = get_vectorizer(stopwords=stopwords)
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
            chunk = chunk.copy()
            chunk['logits'] = logit
            logit_chunks.append(chunk)
            logger.info(f"Processed chunk {idx+1}. Rows processed: {len(chunk)}. Logit: {logit}")
    except Exception as e:
        logger.error(f"Error in logits extraction: {e}")
    
    return logit_chunks
