"""
Constants and configuration values for the project.
"""

import os

CHUNK_SIZE = 100000

# DATA PATHS
PREPARED_DATA = os.path.join("data", "raw_prepared.csv")
PROCESSED_DATA = os.path.join("data", "final_processed.csv")

# TRAIN/TEST splits
ISSUE_TRAIN = os.path.join("data", "issue_train.csv")
ISSUE_TEST = os.path.join("data", "issue_test.csv")
SERIES_TRAIN = os.path.join("data", "series_train.csv")
SERIES_TEST = os.path.join("data", "series_test.csv")
PARA_TRAIN = os.path.join("data", "para_train.csv")
PARA_TEST = os.path.join("data", "para_test.csv")

# LOGITS data paths
ISSUE_LOGITS_DATA = os.path.join("data", "issue_logits.csv")
SERIES_LOGITS_DATA = os.path.join("data", "series_logits.csv")
PARA_LOGITS_DATA = os.path.join("data", "para_logits.csv")

# MAX POOL data paths
ISSUE_MAX_POOL_DATA = os.path.join("data", "issue_max_pool.csv")
SERIES_MAX_POOL_DATA = os.path.join("data", "series_max_pool.csv")
PARA_MAX_POOL_DATA = os.path.join("data", "para_max_pool.csv")

# PROBABILITY data paths
ISSUE_PROB_DATA = os.path.join("data", "issue_probabilities.csv")
SERIES_PROB_DATA = os.path.join("data", "series_probabilities.csv")
PARA_PROB_DATA = os.path.join("data", "para_probabilities.csv")

# TRAINED MODEL paths
ISSUE_TRAINED_MODEL = os.path.join("models", "issue_clf.pkl")
SERIES_TRAINED_MODEL = os.path.join("models", "series_clf.pkl")
PARA_TRAINED_MODEL = os.path.join("models", "para_clf.pkl")

# DOCUMENT LEVEL data paths
DOC_PATH = os.getenv("DOC_LEVEL_DIR", "doc_level")
DOC_ISSUE_DATA = os.path.join(DOC_PATH, "doc_issues.csv")
DOC_SERIES_DATA = os.path.join(DOC_PATH, "doc_series.csv")
DOC_ISSUE_LOGITS = os.path.join(DOC_PATH, "doc_issues_logits.csv")
DOC_SERIES_LOGITS = os.path.join(DOC_PATH, "doc_series_logits.csv")
DOC_ISSUE_PROB = os.path.join(DOC_PATH, "doc_issues_prob.csv")
DOC_SERIES_PROB = os.path.join(DOC_PATH, "doc_series_prob.csv")
DOC_ISSUE_TRAIN = os.path.join(DOC_PATH, "doc_issues_train.csv")
DOC_ISSUE_TEST = os.path.join(DOC_PATH, "doc_issues_test.csv")
DOC_SERIES_TRAIN = os.path.join(DOC_PATH, "doc_series_train.csv")
DOC_SERIES_TEST = os.path.join(DOC_PATH, "doc_series_test.csv")
DOC_ISSUE_MODEL = os.path.join(DOC_PATH, "doc_issues_clf.pkl")
DOC_SERIES_MODEL = os.path.join(DOC_PATH, "doc_series_clf.pkl")