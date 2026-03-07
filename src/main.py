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
        choices=["etl"]
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

def main():
    args = parse_args()
    cfg = _load_base_cfg()

    task = args.task
    year = args.year

    try:
        if task == "etl":
            assert year is not None, "--year is required for etl"
            run_etl(year=year, cfg=cfg)
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
