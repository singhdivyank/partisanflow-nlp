"""Global constants derived from config and hard-coded sentinel values."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"

BASE_CONFIG_PATH: Path = CONFIG_DIR / "base_config.yaml"
SPARK_CONFIG_PATH: Path = CONFIG_DIR / "spark_config.yaml"
MODEL_CONFIG_PATH: Path = CONFIG_DIR / "model_config.yaml"
AIRFLOW_CONFIG_PATH: Path = CONFIG_DIR / "airflow_config.yaml"

# Schema
# Raw parquet columns
COL_SERIES   = "series"
COL_ISSUE    = "issue"
COL_DATE     = "date"
COL_TEXT     = "text"
COL_YEAR     = "year"

# Metadata / label columns
COL_CONTENTS     = "contents"
COL_RAW_LABEL    = "label"

# Processed / feature columns
COL_PARAGRAPH_ID  = "paragraph_id"
COL_PARAGRAPHS    = "paragraphs"
COL_CLEANED       = "cleaned_paragraph"
COL_WORDS         = "words"
COL_RAW_FEATURES  = "rawFeatures"
COL_FEATURES      = "features"
COL_INGESTION_TS  = "ingestion_ts"
COL_FEATURE_TS    = "feature_ts"

# Label values
LABEL_DEMOCRATIC  = 0.0
LABEL_REPUBLICAN  = 1.0
LABEL_INDEPENDENT = 2.0
LABEL_OTHER       = 3.0
TRAIN_LABELS      = [LABEL_DEMOCRATIC, LABEL_REPUBLICAN]

# Prediction columns
COL_PRED_LABEL    = "pred_label"
COL_PREDICTION    = "prediction"
COL_PROB_0        = "prob_democrat"
COL_PROB_1        = "prob_republican"
COL_PROBABILITY   = "probability"
COL_RAW_PRED      = "rawPrediction"
COL_MODEL_NAME    = "model_name"
COL_MODEL_VERSION = "model_version"
COL_PRED_TS       = "prediction_ts"
COL_RUN_ID        = "run_id"

# Drift columns
COL_REF_YEAR      = "ref_year"
COL_METRIC_NAME   = "metric_name"
COL_METRIC_VALUE  = "metric_value"
COL_COMPUTED_TS   = "computed_ts"

PARTY_MAP = {
    "democratic":  LABEL_DEMOCRATIC,
    "republican":  LABEL_REPUBLICAN,
    "independent": LABEL_INDEPENDENT,
}

# Excluded series identifiers
EXCLUDED_SERIES = {"xref", "(no report.)"}

# Table / path names
FEATURE_STORE_NAME       = "newspaper_features_v1"
PREDICTIONS_TABLE        = "newspaper_predictions_v1"
EVAL_PREDICTIONS_TABLE   = "predictions_1869_eval_v1"
DRIFT_METRICS_TABLE      = "newspaper_drift_metrics_v1"

# MLflow
MLFLOW_REGISTRY_NAME     = "newspaper_partisanship_classifier"
MLFLOW_EXPERIMENT_NAME   = "newspaper_partisanship"
MLFLOW_STAGE_STAGING     = "Staging"
MLFLOW_STAGE_PRODUCTION  = "Production"

# Model identifiers
MODEL_LR   = "logistic_regression"
MODEL_NB   = "naive_bayes"
MODEL_SVC  = "linear_svc"
ALL_MODELS = [MODEL_LR, MODEL_NB, MODEL_SVC]

# Year ranges
TRAIN_YEAR      = 1869
INFERENCE_YEARS = [1870, 1871, 1872, 1873, 1874]
REFERENCE_YEAR  = 1869

# Drift metric names
DRIFT_PSI_PROB1          = "psi_prob1"
DRIFT_KL_PROB1           = "kl_prob1"
DRIFT_MEAN_PROB1         = "mean_prob1"
DRIFT_VAR_PROB1          = "var_prob1"
DRIFT_HIGH_CONF_FRAC     = "high_confidence_fraction"
DRIFT_PRED_LABEL_DIST    = "pred_label_distribution"

# Drift thresholds
PSI_WARNING_THRESHOLD    = 0.1
PSI_CRITICAL_THRESHOLD   = 0.2
HIGH_CONF_PROB_THRESHOLD = 0.9

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
"yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
"they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
"these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", 
"do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
"of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
"after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
"further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
"few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
"too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
"a", "about", "above", "across", "after", "afterwards", "again", "against", "al", "all", "almost", "alone", 
"along", "already", "also", "although", "always", "am", "among", "amongst", "an", "analyze", "and", "another", 
"any", "anyhow", "anyone", "anything", "anywhere", "applicable", "apply", "are", "around", "as", "assume", "at", 
"be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "being", "below", 
"beside", "besides", "between", "beyond", "both", "but", "by", "came", "cannot", "cc", "cm", "come", "compare", 
"could", "de", "dealing", "department", "depend", "did", "discover", "dl", "do", "does", "during", "each", "ec", 
"ed", "effected", "eg", "either", "else", "elsewhere", "enough", "et", "etc", "ever", "every", "everyone", 
"everything", "everywhere", "except", "find", "for", "found", "from", "further", "get", "give", "go", "gov", 
"had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", 
"him", "himself", "his", "how", "however", "hr", "ie", "if", "ii", "iii", "in", "inc", "incl", "indeed", "into", 
"investigate", "is", "it", "its", "itself", "j", "jour", "journal", "just", "kg", "last", "latter", "latterly", 
"lb", "ld", "letter", "like", "ltd", "made", "make", "many", "may", "me", "meanwhile", "mg", "might", "ml", "mm", 
"mo", "more", "moreover", "most", "mostly", "mr", "much", "must", "my", "myself", "namely", "neither", "never", 
"nevertheless", "next", "no", "nobody", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", 
"on", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", 
"oz", "per", "perhaps", "pm", "precede", "presently", "previously", "pt", "rather", "regarding", "relate", "said", 
"same", "seem", "seemed", "seeming", "seems", "seriously", "several", "she", "should", "show", "showed", "shown", 
"since", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "studied", 
"sub", "such", "take", "tell", "th", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", 
"thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "thorough", "those", "though", 
"through", "throughout", "thru", "thus", "to", "together", "too", "toward", "towards", "try", "type", "ug", "under", 
"unless", "until", "up", "upon", "us", "used", "using", "various", "very", "via", "was", "we", "were", "what", 
"whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", 
"wherever", "whether", "which", "while", "whither", "who", "whoever", "whom", "whose", "why", "will", "with", 
"within", "without", "wk", "would", "wt", "yet", "you", "your", "yours", "yourself", "yourselves", "yr"
]
