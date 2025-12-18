"""
Centralised path configuration. Eliminates duplicate path selection logic across modules.
"""

from typing import Dict

from utils.consts import *

# paragraph level paths
_PARA_PATHS = {
    'issue': {
        'train': ISSUE_TRAIN, 
        'test': ISSUE_TEST, 
        'model': ISSUE_TRAINED_MODEL,
        'logits': ISSUE_LOGITS_DATA, 
        'max_pool': ISSUE_MAX_POOL_DATA, 
        'prob': ISSUE_PROB_DATA
    },
    'series': {
        'train': SERIES_TRAIN, 
        'test': SERIES_TEST, 
        'model': SERIES_TRAINED_MODEL,
        'logits': SERIES_LOGITS_DATA, 
        'max_pool': SERIES_MAX_POOL_DATA, 
        'prob': SERIES_PROB_DATA
    },
    'para': {
        'train': PARA_TRAIN, 
        'test': PARA_TEST, 
        'model': PARA_TRAINED_MODEL,
        'logits': PARA_LOGITS_DATA, 
        'max_pool': PARA_MAX_POOL_DATA, 
        'prob': PARA_PROB_DATA
    }
}

# document level paths
_DOC_PATHS = {
    'issue': {
        'doc_data': DOC_ISSUE_DATA,
        'logits': DOC_ISSUE_LOGITS,
        'train': DOC_ISSUE_TRAIN,
        'test': DOC_ISSUE_TEST,
        'model': DOC_ISSUE_MODEL,
        'prob': DOC_ISSUE_PROB
    },
    'series': {
        'doc_data': DOC_SERIES_DATA,
        'logits': DOC_SERIES_LOGITS,
        'train': DOC_SERIES_TRAIN,
        'test': DOC_SERIES_TEST,
        'model': DOC_SERIES_MODEL,
        'prob': DOC_SERIES_PROB
    }
}

def get_para_paths(split_col: str) -> Dict[str, str]:
    """
    Get file paths for paragraph-level pipeline.

    Parameters:
    split_col (str): Column name to split on for unique values ('issue', 'series', or 'para').

    Returns:
    Dict[str, str]: Dictionary containing file paths for train, test, model, logits, max_pool, and prob data.
    """

    return _PARA_PATHS.get(split_col, _PARA_PATHS['issue']).copy()

def get_doc_paths(split_col: str) -> Dict[str, str]:
    """
    Get file paths for document-level pipeline.

    Parameters:
    split_col (str): Column name to split on for unique values ('issue' or 'series').

    Returns:
    Dict[str, str]: Dictionary containing file paths for doc_data, logits, train, test, model, and prob data.
    """

    return _DOC_PATHS.get(split_col, _DOC_PATHS['issue']).copy()