# Newspaper Partisanship Classifier (1869–1874)

This project builds a production-style, on-cluster ML pipeline to detect shifts in newspaper partisanship from 1869 to 1874 using historical newspaper series stored on Northeastern's Explorer server.

> Tech focus: Apache Spark, Airflow, MLflow, HDFS, Spark ML, Streamlit  
> Data size: 21M+ rows (Parquet)

---

## 1. Project Overview

**Goal**

Train a partisanship classifier on 1869 newspapers and apply it to 1870–1874 to measure how partisan each series becomes over time. The system is designed like an industry-grade batch ML pipeline running entirely on-prem (Explorer).

**Key Questions**

- How does predicted partisanship evolve from 1869 to 1874?  
- Are there noticeable drifts in language, vocabulary, or model confidence over time?  
- How stable is the classifier when applied across years?

---

## 2. Architecture

Pipeline:

Raw Parquet (HDFS)  
↓  
Spark ETL Job  
↓  
Feature Store (Spark table, partitioned by year)  
↓  
Training Pipeline (Spark ML on 1869)  
↓  
Model Tracking & Registry (MLflow)  
↓  
Batch Inference (1870–1874)  
↓  
Drift Detection & Monitoring  
↓  
Streamlit Dashboard

Orchestrated with Airflow as a single DAG (`newspaper_partisanship_pipeline`).

---

## 3. Data

- **Source:** Parquet files in HDFS on Explorer (`raw/newspapers/year=*`).  
- **Metadata:** CSV with `series_id`, `issue_id`, and partisan labels (0/1).  
- **Granularity:** Text is split into paragraphs; each paragraph is a training example.

### Preprocessing

- Join raw text with labels on `(series_id, issue_id)` for 1869.  
- Split articles into paragraphs using double newlines (`"\n\n"`).  
- Filter out paragraphs with fewer than 100 words.  
- Persist cleaned data to `processed/newspapers/year=<year>/`.

---

## 4. Features & Feature Store

Text features are built with Spark ML:

- Tokenization  
- Stop word removal  
- Lowercasing  
- Vectorization (HashingTF or CountVectorizer + IDF)

The resulting **feature store** is a Spark table:

- Name: `newspaper_features_v1`  
- Partitioned by `year`  
- Columns: `series_id, issue_id, year, paragraph_id, features, label (nullable), feature_ts`

1869 partition includes labels; 1870–1874 store unlabeled examples for inference.

---

## 5. Model Training

Training uses only 1869 data from the feature store.

### Models

- Logistic Regression (primary)  
- Naive Bayes  
- LinearSVC

Each model is implemented as a Spark ML Pipeline from raw text to predictions.

### Train/Test Split

- Group-/time-aware split: 80% of issues for training, 20% for testing, avoiding exact duplicates across sets.

### Metrics

- Accuracy  
- Macro/weighted F1  
- ROC-AUC (where probabilities are available)  
- Per-class precision/recall/F1  
- Confusion matrix

Per-example predictions on the test set are saved for analysis.

---

## 6. Experiment Tracking & Model Registry

Using MLflow:

- **For each run, log:**
  - Parameters: model type, feature config, hyperparameters  
  - Metrics: Acc, F1, ROC-AUC, per-class F1  
  - Artifacts: confusion matrix, ROC curve, classification report  
  - Model: full Spark pipeline

- **Model Registry:**
  - Name: `newspaper_partisanship_classifier`  
  - Versions tagged with data/feature store versions  
  - Staging → Production promotion based on validation metrics

The batch inference pipeline always uses the current Production model.

---

## 7. Batch Inference (1870–1874)

For each year 1870–1874:

1. Load processed text and/or features from the feature store.  
2. Load Production model from MLflow registry.  
3. Run batch predictions.  
4. Persist predictions to a partitioned table:

`newspaper_predictions_v1`:

- `series_id, issue_id, paragraph_id, year`  
- `pred_label, prob_0, prob_1`  
- `model_name, model_version, prediction_ts`

These predictions are used to compute year-wise trends and drift.

---

## 8. Drift Detection & Monitoring

**Reference:** 1869 predictions and features.

**Data drift:**

- Compare TF-IDF/feature distributions between each year and 1869.  
- Track vocabulary changes and term frequencies.  
- Monitor probability distribution shifts (`prob_1`).

**Concept drift:**

- Distribution of predicted labels per year.  
- Average model confidence and fraction of high-confidence predictions.

Metrics include:

- KL divergence  
- Population Stability Index (PSI)

All drift metrics are stored in `newspaper_drift_metrics_v1` for use in the dashboard.

---

## 9. Dashboard (Streamlit)

Streamlit app visualizes:

1. **Year-wise partisanship trend**  
   - Fraction of predicted partisan paragraphs per year.  
2. **Drift scores over time**  
   - PSI/KL trends vs 1869.  
3. **Distribution comparison plots**  
   - Histograms/ECDFs of prediction probabilities per year.  
4. **Confidence heatmap**  
   - Year vs probability bins, color-coded by density.  
5. **Model performance metrics**  
   - Comparison of Logistic Regression, Naive Bayes, LinearSVC.  
6. **Probability distribution histograms**  
   - For selected year and model.

---

## 10. Orchestration (Airflow)

DAG: `newspaper_partisanship_pipeline`

Tasks:

1. `etl_1869` – Clean and persist 1869 text.  
2. `features_1869` – Build and store 1869 features.  
3. `train_models_1869` – Train and evaluate models, log to MLflow.  
4. `register_best_model` – Promote selected model to Production.  
5. `batch_predict_year` – Parameterized task for 1870–1874 inference.  
6. `compute_drift_year` – Compute drift metrics per year.  
7. `refresh_dashboard_views` – Update aggregated tables for the dashboard.

All tasks are idempotent: reruns safely recompute outputs per year/model version.

---

## 11. How to Run (High-Level)

1. Configure Spark, Airflow, and MLflow on Explorer.  
2. Run the Airflow DAG `newspaper_partisanship_pipeline` (initially `@once`).  
3. Once the pipeline completes:
   - Inspect MLflow for training runs and selected Production model.  
   - Explore predictions and drift tables in Spark/Hive.  
   - Launch the Streamlit app to visualize trends and drift.

---

## 12. Future Work

- Add explanatory analysis (top partisan vs neutral phrases).  
- Introduce topic modeling to see which topics drive partisan shifts.  
- Implement automatic retraining when data or performance thresholds trigger it.

## Project Structure

```
newspaper-partisanship-ml/
│
├── README.md
├── requirements.txt
├── main.py
├── setup.py
├── .env
├── .gitignore
│
├── config/
│   ├── base_config.yaml
│   ├── spark_config.yaml
│   ├── airflow_config.yaml
│   └── model_config.yaml
│
├── data_contracts/
│   ├── schema_validation.py
│   └── expectations.json
│
├── src/
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── clean_transform.py
│   │   ├── validate.py
│   │   └── partition_writer.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── tfidf_pipeline.py
│   │   └── feature_store.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── evaluate.py
│   │   └── register_model.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── batch_predict.py
│   │   ├── write_predictions.py
│   │   └── model_loader.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── data_drift.py
│   │   ├── concept_drift.py
│   │   ├── metrics_logger.py
│   │   └── alerting.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── logger.py
│   │   └── spark_session.py
│
├── dags/
│   └── newspaper_pipeline_dag.py
│
├── dashboard/
│   ├── app.py
│   ├── charts.py
│   └── queries.py
│
└── tests/
    ├── test_etl.py
    ├── test_features.py
    ├── test_training.py
    └── test_drift.py
```

Download two .jar files onto local device and upload to cluster:
1. `delta-spark_2.13-4.0.0.jar` [download link](https://repo1.maven.org/maven2/io/delta/delta-spark_2.13/4.0.0/)
2. `delta-storage-4.0.0.jar` [download link](https://repo1.maven.org/maven2/io/delta/delta-storage/4.0.0/)


scp delta-spark_2.13-4.0.0.jar <user_name>>@<LOGIN_NODE>:~/spark-jars/
ls -lh ~/spark-jars
Added JAR file:///home/singh.divya/spark-jars/delta-spark_2.13-4.0.0.jar at spark://c0615:32817/jars/delta-spark_2.13-4.0.0.jar with timestamp 1774752878790