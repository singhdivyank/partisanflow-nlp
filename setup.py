"""
Package setup for newspaper-partisanship.

Install in editable mode for development:
    pip install -e .

Install with optional extras:
    pip install -e ".[dev]"       # adds pytest, pytest-mock
    pip install -e ".[dashboard]" # adds streamlit, plotly
    pip install -e ".[airflow]"   # adds apache-airflow + spark provider
"""

from setuptools import setup, find_packages
from pathlib import Path

_HERE = Path(__file__).parent
_LONG_DESC = (_HERE / "README.md").read_text(encoding="utf-8") if (_HERE / "README.md").exists() else ""

_INSTALL_REQUIRES = [
    # "pyspark==4.1.1",
    "mlflow==3.10.1",
    "scikit-learn==1.8.0",
    "numpy==2.4.2",
    "pandas==2.3.3",
    "scipy==1.17.1",
    "matplotlib==3.10.8",
    "pyyaml==6.0.3",
    "python-dotenv==1.2.2",
    # "pyarrow>=14.0.0",
]

_EXTRAS = {
    # Streamlit dashboard
    "dashboard": [
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
    ],
    # Airflow orchestration (install on scheduler / worker nodes only)
    "airflow": [
        "apache-airflow==2.9.0",
        "apache-airflow-providers-apache-spark>=4.9.0",
    ],
    # Development and testing
    "dev": [
        "pytest>=8.2.0",
        "pytest-mock>=3.14.0",
        "pytest-cov>=5.0.0",
        "black>=24.0.0",
        "ruff>=0.4.0",
        "mypy>=1.10.0",
    ],
}

_EXTRAS["all"] = (_EXTRAS["dashboard"] + _EXTRAS["dev"])

setup(
    name="newspaper-partisanship-ml",
    version="1.0.0",
    description=(
        "Production-style batch ML pipeline for tracking shifts in "
        "newspaper partisanship (1869–1874) using Apache Spark and MLflow."
    ),
    long_description=_LONG_DESC,
    long_description_content_type="text/markdown",
    author="ml-team",
    python_requires=">=3.10",

    # Package discovery
    # Finds all packages under src/, dashboard/, dags/, data_contracts/
    packages=find_packages(
        exclude=["tests", "tests.*", "*.egg-info"]
    ),

    # Dependencies
    install_requires=_INSTALL_REQUIRES,
    extras_require=_EXTRAS,

    # Package data
    # Ensures non-Python files bundled inside packages are included
    package_data={
        "": [
            "config/*.yaml",
            "data_contracts/*.json",
        ],
    },
    include_package_data=True,

    # CLI entry points
    entry_points={
        "console_scripts": [
            "partisanship-pipeline=src.main:main",
        ],
    },

    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
