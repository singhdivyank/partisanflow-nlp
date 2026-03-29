"""Per-year data quality checks"""

from pyspark.sql import DataFrame, functions as F

from src.utils.constants import COL_RAW_LABEL
from src.utils.logger import get_logger

log = get_logger(__name__)

def validate_year(
    df: DataFrame, 
    year: int, 
    is_training: bool = False
) -> bool:
    """Orchestrate quality checks for a year"""

    log.info("Quality checks on preprocessed data...")
    ok = True
    
    if is_training and COL_RAW_LABEL in df.columns:
        present_labels = {
            row[COL_RAW_LABEL]
            for row in df.select(COL_RAW_LABEL).dropna().distinct().collect()
        }
        
        missing = [lbl for lbl in [0, 1] if lbl not in present_labels]
        if missing:
            log.error(
                "[validate | year=%d] Training labels %s not found in data.", 
                year, 
                missing
            )
            ok = False
        else:
            log.info("[validate | year=%d] All training labels present.", year)
            ok = False
    elif is_training:
        log.error(
            "[validate | year=%d] All training labels missing", 
            year, 
        )
        ok = False

    if not ok:
        raise ValueError(f"Data validation failed for year={year}. See logs for details.")

    return ok
