"""TF-IDF feature extraction pipeline using Spark ML."""

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    CountVectorizer, 
    HashingTF, 
    IDF,
    RegexTokenizer,
    StopWordsRemover
)

from src.utils.constants import (
    COL_CLEANED,
    COL_RAW_FEATURES, 
    COL_WORDS
)
from src.utils.logger import logging

log = logging.getLogger(__name__)


def build_pipeline(
    vectorizer: str, 
    feature_col: str, 
    min_doc_freq: int, 
    num_features: int
) -> Pipeline:
    """Build a PySpark ML Pipeline"""

    tokenizer = RegexTokenizer(
        input_col=COL_CLEANED,
        output_col=COL_WORDS,
        min_token_length=1,
        pattern=r"\W"
    )

    remover = StopWordsRemover(
        inputCol=COL_WORDS, 
        outputCol="_filtered_words"
    )

    idf = IDF(
        inputCol=COL_RAW_FEATURES, 
        outputCol=feature_col,
        minDocFreq=min_doc_freq
    )

    if vectorizer == 'count_idf':
        cv = CountVectorizer(
            inputCol="_filtered_words",
            outputCol=COL_RAW_FEATURES,
            vocabSize=50000,
            minDF=float(min_doc_freq)
        )
        return Pipeline(stages=[tokenizer, remover, cv, idf])
    
    # use hashing vectorizer by default
    hashing_tf = HashingTF(
        inputCol="_filtered_words",
        outputCol=feature_col,
        numFeatures=num_features
    )
    return Pipeline(stages=[tokenizer, remover, hashing_tf, idf])

def fit_feature_pipeline(train_df: DataFrame, vectorizer_config: dict) -> PipelineModel:
    """Fit a feature pipeline on training dataframe"""

    vectorizer = vectorizer_config.get("vectorizer")

    log.info("Fitting feature pipeline (vectorizer=%s)", vectorizer)
    
    pipeline = build_pipeline(
        vectorizer=vectorizer, 
        feature_col=vectorizer_config.get("feature_column"), 
        min_doc_freq=vectorizer_config.get("min_doc_freq"),
        num_features=vectorizer_config.get("num_features")
    )
    
    model = pipeline.fit(train_df)
    log.info("Feature pipeline fitted successfully")
    
    return model

def apply_feature_pipeline(
    pipeline_model: Pipeline,
    df: DataFrame,
    drop_intermediate: bool = True
) -> DataFrame:
    """Transform a DataFrame using a fitted feature pipeline"""
    
    df = pipeline_model.transform(df)
    if drop_intermediate:
        cols_to_drop = [COL_RAW_FEATURES, COL_WORDS, "_filtered_words"]
        df = df.drop(*[c for c in cols_to_drop if c in df.columns])
    
    return df
