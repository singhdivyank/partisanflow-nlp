"""
Centralised Spark session management.
"""

from pyspark.sql import SparkSession

_spark_session = None

def get_spark_session(app_name: str = "PartisanClassification") -> SparkSession:
    """
    Get or create a Spark session.

    Parameters:
    app_name (str): Name of the Spark application.

    Returns:
    SparkSession: The Spark session instance.
    """
    global _spark_session

    if _spark_session is None:
        _spark_session = SparkSession.builder.appName(app_name).getOrCreate()
    
    return _spark_session

def stop_spark_session() -> None:
    """
    Stop the Spark session if it exists.
    """
    global _spark_session

    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None