"""Feature store: orchestrates feature extraction for training, persisting results to HDFS"""

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, functions as F, types as T

from .tfidf_pipeline import fit_feature_pipeline, apply_feature_pipeline
from src.etl.partition_writer import write_partition
from src.utils.constants import COL_FEATURE_TS, COL_RAW_LABEL
from src.utils.logger import logging

log = logging.getLogger(__name__)

def build_and_store_features(
    processed_df: DataFrame, 
    year: int, 
    feature_store_path: str,
    vectorizer: dict,
    pipeline_path: str
):
    """Fit a feature pipeline, transform it, and write to feature store"""

    log.info("Building features for year=%d", year)
    
    pipeline_model = fit_feature_pipeline(
        train_df=processed_df, 
        vectorizer_config=vectorizer
    )
    feature_df = apply_feature_pipeline(
        pipeline_model=pipeline_model,
        df=processed_df,
        drop_intermediate=True
    )

    write_partition(
        df=feature_df,
        base_path=feature_store_path,
        year=year
    )

    if pipeline_path:
        log.info("Saving feature pipeline to %s", pipeline_path)
        pipeline_model.write().overwrite().save(pipeline_path)

def load_feature_pipeline(pipeline_path: str) -> PipelineModel:
    """Load previously saved feature pipeline from HDFS"""
    log.info("Loading feature pipeline from %s", pipeline_path)
    return PipelineModel.load(pipeline_path)

def apply_and_store_features(
    processed_df: DataFrame, 
    pipeline_model: PipelineModel,
    feature_store_path: str,
    year: int
):
    """
    Apply already-fitted pipeline to an inference year's data 
    and save results in feature store
    """

    log.info("Applying feature pipeline to year=%d", year)

    if COL_RAW_LABEL not in processed_df.columns:
        processed_df = processed_df.withColumn(
            COL_RAW_LABEL, F.lit(None).cast(T.DoubleType())
        )
    
    feature_df = apply_feature_pipeline(pipeline_model=pipeline_model, df=processed_df)
    write_partition(
        df=feature_df, 
        base_path=feature_store_path, 
        year=year,
        ts_col=COL_FEATURE_TS
    )

    log.info("Features stored for year=%d", year)
