import os

CHUNK_SIZE = 100000

PREPARED_DATA = os.path.join("data", "raw_prepared.csv")
PROCESSED_DATA = os.path.join("data", "final_processed.csv")

ISSUE_TRAIN = os.path.join("data", "issue_train.csv")
ISSUE_TEST = os.path.join("data", "issue_test.csv")
SERIES_TRAIN = os.path.join("data", "series_train.csv")
SERIES_TEST = os.path.join("data", "series_test.csv")
PARA_TRAIN = os.path.join("data", "para_train.csv")
PARA_TEST = os.path.join("data", "para_test.csv")

ISSUE_LOGITS_DATA = os.path.join("data", "issue_logits.csv")
SERIES_LOGITS_DATA = os.path.join("data", "series_logits.csv")
PARA_LOGITS_DATA = os.path.join("data", "para_logits.csv")

ISSUE_MAX_POOL_DATA = os.path.join("data", "issue_max_pool.csv")
SERIES_MAX_POOL_DATA = os.path.join("data", "series_max_pool.csv")
PARA_MAX_POOL_DATA = os.path.join("data", "para_max_pool.csv")

ISSUE_PROB_DATA = os.path.join("data", "issue_probabilities.csv")
SERIES_PROB_DATA = os.path.join("data", "series_probabilities.csv")
PARA_PROB_DATA = os.path.join("data", "para_probabilities.csv")

ISSUE_TRAINED_MODEL = os.path.join("model", "issue_clf.pkl")
SERIES_TRAINED_MODEL = os.path.join("model", "series_clf.pkl")
PARA_TRAINED_MODEL = os.path.join("model", "para_clf.pkl")