# Historical Partisan Drift Analysis (1869-1874)

**Quantifying Political Discourse Shifts via Out-of-Core Online Learning**

## Overview

This project analyzes "Partisan Drift" in American political discourse during the Reconstruction era. By training a model on a fixed baseline from 1869 and applying it to the textual data from 1870-1874, the system detects how language associated with specific party identities evolved or decayed over subsequent election cycles.

The system uses an out-of-core learning approach to handle large datasets and is served by an interactive Streamlit dashboard and FastAPI backend.

## Technical Architecture

The project follows a modular, three-tier architecture containerized with Docker:

- **Model Layer**: `SGDClassifier` using Log-Loss (L<sub>2</sub> regularized Logistic Regression) for incremental learning on large dataset and small RAM

- **Feature Engineering**: `HashingVectorizer` with Hashing Trick to maintain stateless, fixed-size feature space across temporal shifts

- **API Layer**: FastAPI provides high-performance inference endpoints and serves model interpretability data

- **UI Layer**: Streamlit dashboard for real-time visualization of partisan probability and drift metrics

### Methodology: The "1869 Anchor"

To measure drift, this project implements **Static Label Injection** strategy-

1. **Baseline Training**: model is trained on historical data from 1869, reaching 76% accuracy

2. **Series Mapping**: Partisan labels from 1869 are treated as "ground truth" and mapped to the same `series_id` from the years 1870-1874

3. **Drift Quantification**: measure the divergence between the model's 1869-based predictions and fixed labels. A high "Drift Score" indicates the language used by a specific series has fundamentally shifted away from its 1869 partisan definition

## Technical Specifications

| Component | Specification |
| --------- | ------------- |
| Learning Paradigm | Out-of-Core Learning |
| Model | SGDClassifier (Log-Loss) |
| Vectorization | HashingVectorizer (Stateless) |
| Drift Metric | Mean Absolute Error (MAE) |
| Containerization | Docker Compose |
| Interpretability | SHAP values |

## Repository Structure

```
partisanflow-nlp/
├── api/
│   ├── Dockerfile
│   ├── main.py          # FastAPI code
│   ├── requirements.txt
│   └── model.pkl
├── app/
│   ├── Dockerfile
│   ├── app.py           # Streamlit code
│   └── requirements.txt
└── docker-compose.yml
```

## Installation and Setup

**Prerequisites**: Docker and Docker Compose

**Quick Start**

1. Clone the repo

```
git clone https://github.com/singhdivyank/partisanflow-nlp.git

cd partisanflow-nlp
```

2. Launch the system

```
docker-compose up --build
```

3. Access the application

- Interactive Dashboard: `http://localhost:8501`

- API Documentation: `http://localhost:8000/docs`

## Engineering Challenges and Solutions

| Challenge | Description | Solution |
| --------- | ----------- | -------- |
| Memory Constraints | Dataset was too large for standard `LogisticRegression` | Implemented `SGDClassifer` with `partial_fit` |
| Vocab Mismatch | Emergence of new terminologies over time | Used `HashingVectorizer` to ensure new words do not break fixed-input requirements |
| Interpretability | Logistic Regression coefficients are hard to visualise in a hashed space | Integrated a SHAP based local explanation module to identify the tokens contributing to "Drift" |

## Future Roadmap

- **Dynamic Calibration**: Implementing temperature scaling to caliberate model confidence over time

- **BERT Integration**: Comparing the baseline SGD model with transformer-based approach for deeper semantic understanding

- **Geospatial Analysis**: Mapping drift scores to specific Congressional districts or regions
