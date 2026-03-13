"""
Text cleaning and transformation. Split raw text into paragraphs, clean, and 
assign paragrapgh_id within each series-issue.
"""

import re

from pyspark.sql import DataFrame, functions as F, types as T


from src.utils.constants import (
    COL_CLEANED,
    COL_PARAGRAPH_ID,
    COL_TEXT,
    COL_YEAR,
    STOPWORDS,
)
from src.utils.logger import logging

log = logging.getLogger(__name__)

def _clean_paragraph(text: str, min_word_len) -> str:
    """clean a single paragraph string"""

    if not text or text.strip():
        return ""
    
    # Remove non-alphabetical characters except commas, spaces, and \n\n
    text = re.sub(r"[^\w,\s\n]", " ", text)
    # Remove extra spaces around commas
    text = re.sub(r"\s*,\s*", ",", text).strip()
    # Remove extra spaces between words
    text = re.sub(r"\s+", " ", text).strip()
    # Remove words upto 3 characters
    text = re.sub(rf"\b\w{{1,{min_word_len}}}\b", "", text).strip()
    # Filter out stopwords
    tokens = [
        w.lower() for w in text.split()
        if w.lower() not in STOPWORDS and w.strip()
    ]
    return " ".join(tokens).strip()

def _make_clean_udf(min_word_len: int):
    def _clean(text: str) -> str:
        return _clean_paragraph(text, STOPWORDS, min_word_len)

    return F.udf(_clean, T.StringType())

def _final_clean(paragraph: str) -> str:
    """Reconstruct paragraph from comma-delimited chunks, stripping blanks"""

    parts = [p.strip() for p in paragraph.split(",") if p.strip()]
    return ", ".join(parts)

def split_into_paragraphs(df: DataFrame, separator: str) -> DataFrame:
    """Explode text column into one row per paragraph"""

    df = df.withColumn("_paragraphs", F.split(F.col(COL_TEXT), separator))
    df = df.withColumn("_raw_para", F.explode(F.col("_paragraphs")))
    df = df.drop("_paragraphs", COL_TEXT)
    df = df.withColumn(COL_PARAGRAPH_ID, F.monotonically_increasing_id())
    df = df.withColumnRenamed("_raw_para", COL_TEXT)
    return df

def clean_paragraphs(
    df: DataFrame, 
    min_word_len: int, 
    min_para_words: int
) -> DataFrame:
    """Apply text cleaning UDF to each paragraph row"""

    log.info(
        "Cleaning paragraphs (min_word_len=%d, min_paragraph_words=%d)",
        min_word_len, min_para_words
    )

    clean_udf = _make_clean_udf(min_word_len)
    df = df.withColumn("_cleaned", clean_udf(F.col(COL_TEXT)))
    df = df.withColumn(COL_CLEANED, _final_clean(F.col("_cleaned")))
    df = df.drop("_cleaned", COL_TEXT)

    df = df.filter(
        F.col(COL_CLEANED).isNotNull()
        & (F.size(F.split(F.col(COL_CLEANED), r"\s+")) > min_para_words)
    )
    
    log.info(
        "Paragraphs retained after cleaning (word count > %d): %d", 
        min_para_words, df.count()
    )
    return df

def transform(
    df: DataFrame, 
    year: int, 
    preprocess_config: dict
) -> DataFrame:
    """Full pipeline for one year's data"""
    
    df = df.withColumn(COL_YEAR, F.lit(year).cast(T.IntegerType()))
    df = split_into_paragraphs(df, separator=preprocess_config["paragraph_separator"])
    df = clean_paragraphs(
        df, 
        min_word_len=preprocess_config["min_word_length"], 
        min_para_words=preprocess_config["min_paragraph_words"]
    )
    return df
