# Bengaluru House Prices

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-data%20processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-machine%20learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange)](https://github.com/AbdallahAhmed149/Bengaluru-House-Prices)

A Python-based exploratory data analysis and machine-learning workflow for Bengaluru residential property prices. The project cleans and transforms the Kaggle Bengaluru house-price dataset, explores relationships between property attributes and price, performs feature selection with LassoCV, and trains a Random Forest regression model.

> **Project status:** This repository contains an analysis script and dataset. It is currently an experimental/offline modeling project rather than a deployed API or production service.

## Overview

Property-price datasets commonly contain inconsistent area formats, missing values, categorical attributes, and substantial outliers. This project provides a reproducible workflow for preparing those records and evaluating a regression model against the `price` target.

The workflow covers:

1. Loading and inspecting the source dataset.
2. Normalizing area, size, and square-footage fields.
3. Imputing missing categorical and numeric values.
4. Exploring correlations, distributions, and location-level price averages.
5. Encoding selected categorical features.
6. Selecting features with cross-validated Lasso regression.
7. Training and evaluating a Random Forest regressor.

## Key Features & Metrics

### Core features

- **Data-quality inspection:** Reports shape, schema, missing values, sample records, and cardinality.
- **Property-data normalization:** Converts square-footage ranges such as `2100 - 2850` to their midpoint and extracts numeric bedroom counts from values such as `3 BHK` and `4 Bedroom`.
- **Exploratory analysis:** Generates correlation heatmaps, box plots, kernel-density plots, and location-level average-price summaries.
- **Regression workflow:** Uses LassoCV for feature selection and Random Forest regression with a held-out test set and 5-fold cross-validation.

### Quantitative information

The current script reports the following measurements at runtime rather than committing fixed benchmark values:

- Dataset shape and missing-value counts.
- Number of unique values per feature.
- Best Lasso regularization parameter (`alpha`).
- Mean 5-fold cross-validation score.
- Training-set score and held-out test-set R² score.

No verified accuracy, latency, coverage, or throughput benchmark is currently stored in the repository. This README intentionally avoids presenting unverified performance claims; run the script on the complete dataset to capture reproducible results for your environment.

## System Architecture

The project follows a linear, notebook-style data-science pipeline implemented in a single Python script:

```text
CSV dataset
    │
    ▼
Data loading and inspection
    │
    ▼
Cleaning and type normalization
    │
    ├── Missing-value imputation
    ├── Numeric outlier clipping
    └── Exploratory visualizations
    │
    ▼
Categorical encoding and feature preparation
    │
    ▼
LassoCV feature selection
    │
    ▼
Train/test split and standardization
    │
    ▼
Random Forest regression
    │
    ├── 5-fold cross-validation
    └── Hold-out R² evaluation
```

### Design principles

- **Reproducible transformations:** Data-cleaning logic is encapsulated in the `convert_sqft_to_float` helper and applied consistently across records.
- **Separation of analysis stages:** Inspection, cleaning, visualization, feature engineering, selection, and evaluation are organized into explicit sections.
- **Validation before interpretation:** Cross-validation and a held-out test set are used to assess model behavior.
- **Pragmatic modeling:** Tree-based regression is used to capture nonlinear relationships between property characteristics and price.

## Tech Stack

| Category | Technology | Purpose |
| --- | --- | --- |
| Language | Python | Analysis and model execution |
| Data processing | pandas, NumPy | Tabular data loading, cleaning, and numerical operations |
| Visualization | Matplotlib, Seaborn | Distribution, correlation, and outlier analysis |
| Machine learning | scikit-learn | Imputation, encoding, scaling, feature selection, validation, and regression |
| Categorical encoding | category_encoders | Target encoding for availability values |
| Dataset | Bengaluru house-price CSV | Residential property attributes and prices |

## Getting Started

### Prerequisites

- Python 3.9 or newer
- `pip`
- A local copy of the repository
- The dataset file `bengaluru_house_prices.csv`

### Installation

Clone the repository and create an isolated virtual environment:

```bash
git clone https://github.com/AbdallahAhmed149/Bengaluru-House-Prices.git
cd Bengaluru-House-Prices

python -m venv .venv
```

Activate the environment:

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn category-encoders
```

### Dataset location

The script currently reads the dataset using a Windows-specific path:

```python
pd.read_csv("Kaggle Bengaluru House Price\\bengaluru_house_prices.csv")
```

Place the CSV at that path, or update the input path in `bengaluru_house_prices.py` to point to the repository file:

```python
pd.read_csv("bengaluru_house_prices.csv")
```

For a portable project setup, prefer a path built with `pathlib`:

```python
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "bengaluru_house_prices.csv"
bengaluru_house_data = pd.read_csv(DATA_PATH)
```

## Usage

Run the analysis script from the repository root:

```bash
python bengaluru_house_prices.py
```

The script prints dataset diagnostics and model scores, and opens interactive plots for:

- Numeric-feature correlations
- Numeric-feature box plots
- Kernel-density distributions before and after clipping
- Availability versus price

The main cleaning helper can also be used independently:

```python
from bengaluru_house_prices import convert_sqft_to_float

print(convert_sqft_to_float("2100 - 2850"))  # 2475.0
print(convert_sqft_to_float("1200"))         # 1200.0
```

### Expected outputs

The console output includes:

```text
Best alpha: <selected regularization value>
<mean 5-fold cross-validation score>
<training-set score>
<held-out test R² score>
```

Exact values depend on the dataset version, preprocessing behavior, installed library versions, and execution environment.

## Reproducibility and Limitations

- The script uses `random_state=42` for the train/test split and cross-validation shuffling.
- Plot rendering requires a graphical environment or a configured Matplotlib backend.
- The source dataset contains inconsistent categorical labels, missing values, mixed units, ranges, and potential outliers.
- The current script is intended for exploratory analysis and should be reviewed before production use.
- Before using the model for business or financial decisions, validate feature encoding, prevent target leakage, establish a formal baseline, and report metrics on an untouched evaluation set.
- A dependency lockfile, automated test suite, CI workflow, API layer, and deployment configuration are not currently included.

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository.
2. Create a focused branch:

   ```bash
   git checkout -b feature/improve-preprocessing
   ```

3. Make the change and document any new assumptions or dependencies.
4. Run the analysis locally and verify that the script completes successfully.
5. Commit with a clear message and push the branch.
6. Open a pull request describing the motivation, implementation, validation steps, and any changes to model behavior.

For larger changes, open an issue first to discuss the proposed approach.

## License

No license file is currently included in the repository. Add a `LICENSE` file before distributing or reusing this project. If you intend to release it under the MIT License, include the standard MIT license text and update this section accordingly.

## Acknowledgements

- Dataset: Kaggle Bengaluru house-price dataset.
- Libraries: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and category_encoders.
