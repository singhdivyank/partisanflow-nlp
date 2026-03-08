from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    NaiveBayes
)

from src.utils.constants import (
    MODEL_LR, 
    MODEL_NB, 
    MODEL_SVC
)

def _build_lr(params: dict) -> LogisticRegression:
    return LogisticRegression(
        featuresCol=params.get("featuresCol"),
        labelCol=params.get("labelCol"),
        predictionCol=params.get("predictionCol"),
        probabilityCol=params.get("probabilityCol"),
        rawPredictionCol=params.get("rawPredictionCol"),
        family=params.get("family"),
        maxIter=params.get("maxIter"),
        regParam=params.get("regParam"),
        elasticNetParam=params.get("elasticNetParam"),
        tol=params.get("tol"),
        fitIntercept=params.get("fitIntercept")
    )

def _build_nb(params: dict) -> NaiveBayes:
    return NaiveBayes(
        featuresCol=params.get("featuresCol"),
        labelCol=params.get("labelCol"),
        predictionCol=params.get("prediction"),
        probabilityCol=params.get("probabilityCol"),
        modelType=params.get("modelType"),
        smoothing=params.get("smoothing")
    )

def _build_svc(params: dict) -> LinearSVC:
    return LinearSVC(
        featuresCol=params.get("featuresCol"),
        labelCol=params.get("labelCol"),
        predictionCol=params.get("predictionCol"),
        maxIter=params.get("maxIter"),
        regParam=params.get("regParam"),
        tol=params.get("tol"),
        fitIntercept=params.get("fitIntercept")
    )

_BUILDER_MAP = {
    MODEL_LR: _build_lr,
    MODEL_NB: _build_nb,
    MODEL_SVC: _build_svc
}