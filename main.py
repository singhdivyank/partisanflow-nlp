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

def run_etl(year: int, cfg: dict, use_delta: bool = True) -> None:
    """
    Run ETL pipeline with optional Delta Lake support
    """
    
    from data_contracts.schema_validation import validate_raw, validate_year
    from src.etl.clean_transform import transform
    from src.etl.ingest import ingest

    spark = get_spark_session()
    paths = cfg["paths"]

    log.info("ETL: year=%d", year)

    raw_df = ingest(
        spark=spark,
        parquet_path=os.getenv("PARQUET_FILE_PATH"),
        metadata_path=os.getenv("METADATA_PATH"),
        label_col=cfg['label_column_metadata'],
        label_delimiter=cfg['label_delimiter'],
        labels=cfg['labels'],
        year=year
    )
    
    validated = validate_raw(raw_df, raise_on_error=True)
    if not validated:
        log.warning("Raw data invalidated")
        return
    
    processed_df = transform(
        df=raw_df, 
        year=year, 
        preprocess_config=cfg["preprocessing"]
    )
    year_validated = validate_year(
        df=processed_df, 
        year=year, 
        is_training=(year == TRAIN_YEAR)
    )
    if not year_validated:
        log.warning("Year validation failed")
        return
    
    log.info("Validation successfully completed. No. of issues: %d", 0)

    if use_delta:
        from src.etl.partition_writer import write_to_delta

        delta_base = os.path.expanduser("~/delta")
        delta_processed_path = os.path.join(delta_base, "processed_data")

        log.info("Writing processed data to Delta Lake: %s", delta_processed_path)
        
        writer = processed_df.write
        write_to_delta(
            writer=writer,
            path=delta_processed_path,
        )
    else:
        from src.etl.partition_writer import write_partition

        write_partition(
            df=processed_df, 
            base_path=paths["processed_base"], 
            year=year
        )
    
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
    
    df = read_partition(
        spark=spark,
        base_path=paths["processed_base"], 
        year=year
    )

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

    spark = get_spark_session()
    paths = cfg["paths"]
    split_cfg = cfg["split"]

    features_df = read_partition(
        spark=spark, 
        base_path=paths["feature_store"], 
        year=TRAIN_YEAR
    )
    train_df, test_df = perform_split(features_df, split_cfg)
    results = train_all_models(model_cfg=model_cfg, train_df=train_df, test_df=test_df)
    log.info("Training complete")
    return results

def run_register(model_cfg: dict) -> None:
    import mlflow
    
    from src.utils.constants import (
        MLFLOW_REGISTRY_NAME, 
        MLFLOW_STAGE_PRODUCTION, 
        MLFLOW_STAGE_STAGING
    )

    mlflow_uri = model_cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    staging = client.get_latest_versions(MLFLOW_REGISTRY_NAME, stages=[MLFLOW_STAGE_STAGING])
    if not staging:
        log.warning("No Staging model found - nothing to promote")
        return
    
    version = staging[0].version
    log.info("Promoting Staging version %s to Production", version)
    
    client.transition_model_version_stage(
        name=MLFLOW_REGISTRY_NAME,
        version=version,
        stage=MLFLOW_STAGE_PRODUCTION,
        archive_existing_versions=True,
    )
    
    log.info("Model version %s transitioned to %s.", version, MLFLOW_STAGE_PRODUCTION)

def run_predict(year: int, cfg: dict, model_cfg) -> None:
    from src.etl.partition_writer import read_partition, write_partition
    from src.inference.batch_predict import load_production_model, run_predictions
    from src.utils.constants import COL_PRED_TS, COL_YEAR

    spark = get_spark_session()
    paths = cfg["paths"]
    mlflow_uri = model_cfg["mlflow"]["tracking_uri"]

    log.info("PREDICT: year=%d", year)

    feature_df = read_partition(
        spark=spark, 
        base_path=paths["feature_store"], 
        year=TRAIN_YEAR
    )
    model, version = load_production_model(tracking_uri=mlflow_uri)
    predictions_df = run_predictions(
        model=model,
        feature_df=feature_df,
        model_version=version,
        year=year,
        has_probability=True
    )
    log.info("Writing predictions for year=%d to %s", year, paths["predictions"])
    write_partition(
        df=predictions_df, 
        base_path=paths["predictions"], 
        year=year,
        ts_col=COL_PRED_TS,
        partition_cols=[COL_YEAR],
        mode="overwrite"
    )
    log.info("Predictions written for year=%d", year)

def run_drift(cfg: dict, year: int) -> None:
    from src.monitoring.comp_drift import (
        compute_concept_drift, 
        compute_data_drift, 
        write_drift_metrics
    )
    from src.utils.constants import REFERENCE_YEAR

    spark = get_spark_session()
    paths = cfg["paths"]

    log.info("DRIFT: year=%d", year)

    ref_df = spark.read.parquet(f"{paths['predictions']}/year={REFERENCE_YEAR}")
    cur_df = spark.read.parquet(f"{paths['predictions']}/year={year}")

    data_drift = compute_data_drift(
        ref_df=ref_df,
        cur_df=cur_df,
        year=year,
        model_version="Production",
    )
    concept_drift = compute_concept_drift(
        ref_df=ref_df,
        cur_df=cur_df,
        year=year,
        model_version="Production"
    )

    drift_metrics = data_drift + concept_drift
    write_drift_metrics(spark=spark, metrics=drift_metrics, paths=paths["drift_metrics"])
    log.info("Drift metrics computed for year=%d", year)

def parse_args():
    parser = argparse.ArgumentParser(description="Newspaper Partisanship ML Pipeline")
    parser.add_argument(
        "--task",
        required=True,
        choices=["etl", "features", "train", "register", "predict", "drift"]
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument(
        "--use-delta",
        dest="use_delta",
        action="store_true",
        help="Enable Delta Lake usage"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = _load_base_cfg()
    model_cfg = _load_model_cfg()

    task = args.task
    year = args.year
    use_delta = args.use_delta

    try:
        if task == "etl":
            assert year is not None, "--year is required for etl"
            run_etl(year=year, cfg=cfg, use_delta=use_delta)
        elif task == "features":
            assert year is not None, "--year is required for features"
            run_features(year=year, cfg=cfg)
        elif task == "train":
            results = run_train(cfg=cfg, model_cfg=model_cfg)
            print(results)
        elif task == "register":
            run_register(model_cfg=model_cfg)
        elif task == "predict":
            run_predict(year=year, cfg=cfg, model_cfg=model_cfg)
        elif task == "drift":
            run_drift(cfg=cfg, year=year)
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
