"""Pipeline entry point"""

import argparse
import os
import sys
from dotenv import load_dotenv

import yaml

from src.utils.constants import (
    BASE_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_YEAR
)
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session, stop_spark_session

log = get_logger(__name__)
env = load_dotenv('.env')


def _load_base_cfg() -> dict:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _load_model_cfg() -> dict:
    with open(MODEL_CONFIG_PATH) as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Newspaper Partisanship ML Pipeline")
    parser.add_argument(
        "--task",
        required=True,
        choices=["etl", "features", "train"]
    )
    parser.add_argument("--year", type=int, default=None)
    return parser.parse_args()

def run_etl(year: int, cfg: dict) -> None:
    
    from data_contracts.schema_validation import validate_raw
    from src.etl.clean_transform import transform
    from src.etl.ingest import ingest
    from src.etl.partition_writer import write_partition
    from src.etl.validate import validate_year

    spark = get_spark_session()
    paths = cfg["paths"]

    log.info("ETL: year=%d", year)

    raw_df = ingest(
        spark=spark,
        parquet_path=os.getenv("PARQUET_FILE_PATH"),
        metadata_path=os.getenv("METADATA_PATH")
    )
    
    validate_raw(raw_df, raise_on_error=True)
    processed_df = transform(df=raw_df, year=year)
    validate_year(
        df=processed_df, 
        year=year, 
        is_training=(year == TRAIN_YEAR)
    )
    
    write_partition(df=processed_df, base_path=paths["processed_base"], year=year)
    log.info("ETL complete for year=%d", year)

def run_features(year: int, cfg: dict) -> None:
    
    from src.etl.partition_writer import read_partition
    from src.features.feature_store import (
        apply_and_store_features,
        build_and_store_features,
        load_feature_pipeline
    )

    spark = get_spark_session()
    paths = cfg["paths"]

    pipeline_save_path = paths["feature_store"] + "/pipeline"

    log.info("Generating features for year=%d", year)
    
    df = read_partition(spark=spark, base_path=paths["processed_base"], year=year)

    if year == TRAIN_YEAR:
        build_and_store_features(
            processed_df=df, 
            year=TRAIN_YEAR,
            feature_store_path=paths["feature_store"], 
            vectorizer=cfg["features"]["vectorizer"], 
            pipeline_path=pipeline_save_path
        )
    else:
        pipeline_model = load_feature_pipeline(pipeline_path=pipeline_save_path)
        apply_and_store_features(
            processed_df=df, 
            pipeline_model=pipeline_model,
            feature_store_path=paths["feature_store"],
            year=year
        )

    log.info("Feature extraction complete for year=%d", year)

def run_train(cfg: dict, model_cfg: dict) -> dict:
    from src.etl.partition_writer import read_partition
    from src.training.train import train_all_models, perform_split

    spark=get_spark_session()
    paths = cfg["paths"]
    split_cfg = cfg["split"]

    features_df = read_partition(spark=spark, base_path=paths["feature_store"], year=TRAIN_YEAR)
    train_df, test_df = perform_split(features_df, split_cfg)
    results = train_all_models(model_cfg=model_cfg, train_df=train_df, test_df=test_df)
    log.info("Training complete")
    return results

def main():
    args = parse_args()
    cfg = _load_base_cfg()
    model_cfg = _load_model_cfg()

    task = args.task
    year = args.year

    try:
        if task == "etl":
            assert year is not None, "--year is required for etl"
            run_etl(year=year, cfg=cfg)
        elif task == "features":
            assert year is not None, "--year is required for features"
            run_features(year=year, cfg=cfg)
        elif task == "train":
            run_train(cfg=cfg, model_cfg=model_cfg)
        else:
            log.error("Unknown task: %s", task)
            sys.exit(1)
    except Exception:
        log.exception("Pipeline task '%s' failed", task)
        sys.exit(1)
    finally:
        spark = get_spark_session.__wrapped__() if hasattr(get_spark_session, "__wrapped__") else None
        stop_spark_session(spark)


if __name__ == '__main__':
    main()
