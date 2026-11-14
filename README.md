# Credit Card Fraud Detection

An end-to-end Machine Learning pipeline and Streamlit web application designed to detect fraudulent credit card transactions in high-volume, highly imbalanced transaction streams.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Performance](#model-performance)
- [Running the Project Locally](#running-the-project-locally)
- [Using the Application](#using-the-application)
- [Prediction Pipeline and Input Schema](#prediction-pipeline-and-input-schema)
- [Model Artifacts](#model-artifacts)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview

Credit card fraud poses a critical challenge for global financial systems, resulting in billions of dollars in unauthorized transactions and administrative overhead each year. Detecting fraudulent behavior in real time requires machine learning systems that can distinguish subtle patterns of compromise from ordinary cardholder activity.

A defining characteristic of transaction fraud detection is extreme class imbalance: genuine transactions account for over 99% of total transaction volume. Under these conditions, conventional accuracy is an uninformative metric because a trivial classifier predicting all transactions as legitimate easily achieves 99% accuracy while detecting zero fraud instances. 

This project addresses these challenges by building an end-to-end, reproducible machine learning pipeline centered on:
- Automated feature engineering (geospatial Haversine customer-merchant distance, demographic age extraction).
- Class imbalance mitigation via Synthetic Minority Over-sampling Technique (SMOTE).
- Mini-batch stochastic gradient descent classification using log-loss optimization (`SGDClassifier`).
- Evaluation focused on Precision, Recall, and Precision-Recall Area Under Curve (PR-AUC).
- An interactive Streamlit dashboard (`CreditGuard`) supporting single-transaction scoring and batch CSV evaluation with configurable decision thresholds.

---

## Key Features

- **Automated Feature Extraction:** Automatically parses transaction timestamps, derives cardholder age, computes spherical customer-to-merchant distances via the Haversine formula, and prunes uninformative personal identifiers.
- **Leakage-Free Resampling Pipeline:** Encapsulates preprocessing and SMOTE within Scikit-Learn / Imbalanced-Learn pipelines, ensuring synthetic oversampling is restricted strictly to training folds and bypassed during inference.
- **Linear Mini-Batch Classification:** Utilizes `SGDClassifier(loss="log_loss")` to achieve rapid convergence across large-scale datasets while supporting probability generation.
- **Configurable Decision Thresholding:** Exposes an adjustable classification threshold parameter (default `0.50`), allowing risk analysts to shift operational priorities toward higher recall (e.g., threshold `0.3894`).
- **Input Schema Validation:** Validates incoming payloads for column presence, missing values, non-negative amounts, date chronology, and geographic coordinate limits.
- **Interactive Streamlit Web Dashboard:** Provides an interface for monitoring benchmark metrics, scoring individual transactions with risk visualizers, and scoring batch CSV uploads with summary analytics and exportable reports.

---

## Technology Stack

| Category | Technologies |
|:---|:---|
| **Programming Language** | Python 3.10 |
| **Data Processing & Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Imbalanced Learning** | Imbalanced-learn (SMOTE) |
| **Scientific Computing** | SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Application Framework** | Streamlit |
| **Model Serialization** | Joblib |

---

## Dataset

The project utilizes simulated credit card transaction datasets covering genuine and fraudulent operations across various retail merchants, customer demographics, and transaction categories.

### Dataset Dimensions and Class Distribution

| Split | File Location | Records | Legitimate | Fraudulent | Fraud Prevalence |
|:---|:---|---:|---:|---:|---:|
| **Training** | `data/archive (5)/fraudTrain.csv` | 1,296,675 | 1,289,169 | 7,506 | ~0.58% |
| **Testing** | `data/archive (5)/fraudTest.csv` | 555,719 | 553,574 | 2,145 | ~0.38% |

### Key Dataset Characteristics
- **Temporal Nature:** The training dataset spans transactions from January 1, 2019 to June 21, 2020. The test dataset spans transactions from June 21, 2020 to December 31, 2020. The evaluation split is strictly chronological (out-of-time).
- **Target Variable:** `is_fraud` (Binary indicator: `0` = Legitimate transaction, `1` = Fraudulent transaction).
- **Class Imbalance:** For every fraudulent transaction in the training set, there are approximately 172 legitimate transactions. In the test set, there are approximately 258 legitimate transactions per fraud instance.

---

## Project Architecture

```text
Machine Learning -Credit Card Fraud Detection/
├── app.py                          # Streamlit interactive web dashboard
├── requirements.txt                # Python package dependencies
├── README.md                       # Project documentation
├── PROJECT_TECHNICAL_HANDOFF.md    # Technical handoff & audit specification
├── data/
│   ├── archive (5)/
│   │   ├── fraudTrain.csv          # Raw training dataset (1,296,675 rows)
│   │   └── fraudTest.csv           # Raw out-of-time test dataset (555,719 rows)
│   └── processed/
│       ├── val_data.csv            # Stratified validation split dataset
│       └── chronological_val_data.csv # Chronological validation split dataset
├── models/
│   ├── best_model.pkl              # Production baseline pipeline artifact (702 KB)
│   ├── chronological_model.pkl     # Chronological split model artifact (698 KB)
│   ├── tuned_sgd_model.pkl         # Experimental tuned SGD weights
│   └── tuned_sgd_parameters.pkl    # Experimental tuned hyperparameters
├── notebooks/                      # Research and exploratory Jupyter notebooks
│   ├── eda.ipynb                   # Exploratory data analysis
│   ├── data_preprocessing.ipynb    # Feature engineering & scaling experiments
│   ├── imbalance_handling.ipynb    # Resampling benchmarks (SMOTE, Under, Over)
│   ├── model_training.ipynb        # Baseline classifier comparisons
│   └── model_tuning.ipynb          # Hyperparameter tuning benchmarks
├── reports/                        # Performance reports and visualizations
│   ├── confusion_matrix_val.png    # Validation confusion matrix plot
│   ├── confusion_matrix_test.png   # Test confusion matrix plot
│   ├── evaluation_summary.csv      # Validation and test summary metrics
│   ├── model_comparison.csv        # Notebook resampling experiment records
│   └── hyperparameter_tuning_results.csv # Cross-validation search results
└── src/                            # Modular production source code
    ├── __init__.py                 # Package initializer
    ├── feature_engineering.py      # Feature derivation & Haversine distance
    ├── preprocessing.py            # ColumnTransformer configurations
    ├── train.py                    # Pipeline assembly & model training workflows
    ├── evaluate.py                 # Evaluation reports & confusion matrix generation
    └── predict.py                  # Schema validation & inference endpoint
```

### Module Responsibilities

- **`src.feature_engineering`**: Contains `prepare_features()`, which computes customer age at transaction time, derives spherical customer-merchant distance via the Haversine formula, extracts datetime parts, and removes uninformative identifiers.
- **`src.preprocessing`**: Houses `create_preprocessor()` and `create_improved_preprocessor()`, configuring `ColumnTransformer` for numerical standard scaling and categorical one-hot encoding.
- **`src.train`**: Connects feature extraction, column preprocessing, training-only SMOTE resampling, and classifier fitting inside an `imblearn.pipeline.Pipeline`. Supports both stratified and chronological training splits.
- **`src.evaluate`**: Evaluates model pipelines on validation and untouched test sets, computing classification reports, confusion matrices, and exporting summary tables.
- **`src.predict`**: Provides `predict_transaction()`, supporting dictionary, pandas Series, and DataFrame inputs with schema validation and configurable classification thresholds.

---

## Machine Learning Workflow

The implemented machine learning pipeline follows an 8-stage sequence:

```text
Raw Transaction Payload
    │
    ▼
1. Data Ingestion & Exploration
   - Verified 0 null values, 0 duplicate records across 1.29M rows.
    │
    ▼
2. Feature Engineering (src/feature_engineering.py)
   - Haversine customer-to-merchant distance:
     d = 2 * 6371 * arcsin(sqrt(sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)))
   - Customer age calculation:
     age = floor((trans_date_trans_time - dob) / 365.25 days)
   - Datetime component extraction: hour, day, month, day_of_week
   - Pruning of redundant fields: cc_num, first, last, street, city, zip, trans_num, unix_time, lat, long, merch_lat, merch_long
    │
    ▼
3. Train / Validation Splitting
   - Stratified 80/20 split on training data (1,037,340 train / 259,335 val).
   - Test dataset (555,719 rows) remains completely isolated.
    │
    ▼
4. Preprocessing & Encoding (src/preprocessing.py)
   - StandardScaler applied to 4 numerical features:
     ['amt', 'city_pop', 'age', 'customer_merchant_distance_km']
   - OneHotEncoder(handle_unknown='ignore') applied to 5 categorical features:
     ['merchant' (693), 'category' (14), 'gender' (2), 'state' (51), 'job' (494)]
   - Note: Datetime components are omitted by ColumnTransformer(remainder='drop') in baseline.
   - Total transformed feature dimensionality: 1,258 features.
    │
    ▼
5. Class Imbalance Handling (Training Fold Only)
   - SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=42)
   - Expands minority training frauds from 6,006 to 103,134 synthetic samples.
   - Bypassed automatically during inference.
    │
    ▼
6. Model Training (src/train.py)
   - SGDClassifier(loss='log_loss', max_iter=1000, tol=0.001, random_state=42)
   - Fits weights via mini-batch stochastic gradient descent.
    │
    ▼
7. Out-of-Time Model Evaluation (src/evaluate.py)
   - Performance measured on untouched fraudTest.csv using PR-AUC, F1, Recall, Precision.
    │
    ▼
8. Fraud Prediction Interface (src/predict.py)
   - Input schema validation -> Preprocessing -> Classification -> Probability & Class output.
```

---

## Model Performance

The model was evaluated on both the out-of-sample validation fold and the final out-of-time test dataset (`fraudTest.csv`).

### Final Test Performance (Out-of-Time Dataset)

| Metric | Test Result |
|:---|---:|
| **Accuracy** | 99.26% |
| **Precision** | 25.90% |
| **Recall** | 48.81% |
| **F1-Score** | 33.85% |
| **ROC-AUC** | 85.82% |
| **PR-AUC (Average Precision)** | **15.49%** |

### Final Test Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|:---|---:|---:|
| **Actual Legitimate** | 550,579 | 2,995 |
| **Actual Fraud** | 1,098 | 1,047 |

### Evaluation Interpretation for Fraud Detection

- **Why Accuracy is Insufficient:** Overall accuracy is 99.26%. However, because only 0.386% of test transactions are fraudulent, a model that labels every transaction as legitimate would achieve 99.61% accuracy while catching 0 frauds. Accuracy masks severe detection failures in imbalanced classification.
- **Recall (48.81%):** The model detects 1,047 out of 2,145 total fraudulent transactions at the default 0.50 threshold, missing 1,098 frauds (False Negatives).
- **Precision (25.90%):** When the model flags a transaction as fraud, it is correct in approximately 1 out of 4 cases, generating 2,995 false alarms across 553,574 legitimate transactions.
- **PR-AUC (15.49%):** Precision-Recall Area Under Curve evaluates trade-offs across all possible decision thresholds, providing a more reliable measure of model discrimination than ROC-AUC when the positive class is rare.

### Validation vs. Final Test Performance Comparison

The validation split used in the baseline experiment was a random stratified split of `fraudTrain.csv`, whereas the final test set is strictly chronological. The metrics reflect this distribution shift:

| Metric | Validation Dataset (`val_data.csv`) | Final Test Dataset (`fraudTest.csv`) |
|:---|---:|---:|
| **Sample Count** | 259,335 | 555,719 |
| **Fraud Prevalence** | 0.58% (1,501 cases) | 0.38% (2,145 cases) |
| **Accuracy** | 99.21% | 99.26% |
| **Precision** | 37.53% | 25.90% |
| **Recall** | 54.16% | 48.81% |
| **F1-Score** | 44.34% | 33.85% |
| **ROC-AUC** | 89.72% | 85.82% |
| **PR-AUC** | **24.72%** | **15.49%** |

*Validation Confusion Matrix:* True Negatives: `256,481` | False Positives: `1,353` | False Negatives: `688` | True Positives: `813`.

---

## Running the Project Locally

### Prerequisites
- Python 3.10 or higher
- Git

### Setup Instructions

#### macOS and Linux
```bash
git clone https://github.com/rahul15-manch/Humanli-Assignment.git
cd Humanli-Assignment

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

#### Windows
```cmd
git clone https://github.com/rahul15-manch/Humanli-Assignment.git
cd Humanli-Assignment

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Running Pipeline Scripts Directly

```bash
# Execute model training (saves to models/best_model.pkl)
python -m src.train

# Execute pipeline evaluation and generate reports
python -m src.evaluate

# Run sample CLI prediction test
python -m src.predict
```

---

## Using the Application

The Streamlit web application provides four primary views accessible from the sidebar:

### 1. Dashboard
- Displays active model status and architecture specifications.
- Presents verified historical evaluation metrics for validation and test datasets.
- Summarizes dataset size, fraud distribution, and benchmark performance.

### 2. Single Transaction Prediction
- Enter transaction data through structured input controls (dropdowns for category, state, and gender; date and time pickers for transaction time and date of birth; numeric controls for amounts, populations, and coordinates).
- Adjust the classification decision threshold slider (default `0.50`, recommended `0.3894`).
- Click **Analyze Transaction** to execute the pipeline.
- View the classification outcome (`Legitimate Transaction` or `Fraudulent Transaction`), risk probability percentage, and inspect submitted payload data.

### 3. Batch CSV Prediction
- Download a sample template CSV (`creditguard_sample_template.csv`) to review the required column structure.
- Upload any CSV file containing the 13 mandatory transaction columns.
- Adjust the batch decision threshold slider.
- Click **Run Batch Prediction** to score all transactions.
- Review aggregate scoring metrics (total records, legitimate count, flagged fraud count, average fraud score), inspect high-risk transactions, and download the full scored CSV.

### 4. Model Information
- Review the complete pipeline flow diagram, feature space dimensions (1,258 features), and documented technical limitations.

---

## Prediction Pipeline and Input Schema

The prediction API is exposed via `src.predict.predict_transaction()`:

```python
from src.predict import predict_transaction

# Single transaction (dict, Series, or 1-row DataFrame)
result_label, fraud_probability = predict_transaction(transaction_dict, threshold=0.5)

# Batch DataFrame
results_df = predict_transaction(batch_dataframe, threshold=0.5)
```

### Mandatory Input Fields

| Field Name | Expected Type | Format / Constraints | Description |
|:---|:---|:---|:---|
| `trans_date_trans_time` | string or datetime | `YYYY-MM-DD HH:MM:SS` | Transaction timestamp |
| `merchant` | string | String identifier | Merchant name |
| `category` | string | 1 of 14 valid categories | Spending category |
| `amt` | numeric | Float $\ge 0.0$ | Transaction dollar amount |
| `gender` | string | `'M'` or `'F'` | Cardholder gender |
| `state` | string | 2-letter US state code | Cardholder residential state |
| `city_pop` | numeric | Integer $\ge 0$ | Residential city population |
| `job` | string | String identifier | Cardholder occupation |
| `dob` | string or date | `YYYY-MM-DD` | Cardholder date of birth |
| `lat` | numeric | Float between `-90.0` and `90.0` | Customer home latitude |
| `long` | numeric | Float between `-180.0` and `180.0` | Customer home longitude |
| `merch_lat` | numeric | Float between `-90.0` and `90.0` | Merchant latitude |
| `merch_long` | numeric | Float between `-180.0` and `180.0` | Merchant longitude |

*Optional / Ignored Fields:* `Unnamed: 0`, `cc_num`, `first`, `last`, `street`, `city`, `zip`, `trans_num`, `unix_time`, `is_fraud`. These are dropped automatically if present.

### Validation Behavior
- **Missing Columns:** Raises `ValueError: Missing required columns: [...]`
- **Null Values:** Raises `ValueError: Null values found in required columns: [...]`
- **Negative Values:** Raises `ValueError` for negative `amt` or `city_pop`.
- **Coordinate Limits:** Raises `ValueError` if coordinates fall outside valid latitude/longitude ranges.
- **Date Inconsistency:** Raises `ValueError` if `dob` occurs after `trans_date_trans_time`.
- **Unseen Categories:** Categories not present during training are encoded as all-zero dummy variables without throwing errors (`OneHotEncoder(handle_unknown='ignore')`).

---

## Model Artifacts

| Artifact Path | Size | Description |
|:---|---:|:---|
| [`models/best_model.pkl`](file:///Users/rahulmanchanda/Desktop/Machine%20Learning%20-Credit%20Card%20Fraud%20Detection%20/models/best_model.pkl) | 702 KB | Active baseline pipeline artifact (`imblearn.pipeline.Pipeline`). Ingests raw transactions directly and produces fraud predictions. |
| [`models/chronological_model.pkl`](file:///Users/rahulmanchanda/Desktop/Machine%20Learning%20-Credit%20Card%20Fraud%20Detection%20/models/chronological_model.pkl) | 698 KB | Model artifact trained on a chronological train-validation split (cutoff `2020-03-06`). |
| [`models/tuned_sgd_model.pkl`](file:///Users/rahulmanchanda/Desktop/Machine%20Learning%20-Credit%20Card%20Fraud%20Detection%20/models/tuned_sgd_model.pkl) | 11 KB | Classifier weights from experimental RandomizedSearchCV tuning in notebook. |
| [`models/tuned_sgd_parameters.pkl`](file:///Users/rahulmanchanda/Desktop/Machine%20Learning%20-Credit%20Card%20Fraud%20Detection%20/models/tuned_sgd_parameters.pkl) | 164 B | Serialized parameter dictionary identified during hyperparameter search. |

*Inference Requirement:* Loading `best_model.pkl` via `joblib.load()` requires that `src` is in the Python search path because the pipeline embeds a `FunctionTransformer` pointing to `src.feature_engineering.prepare_features`.

---

## Limitations and Future Improvements

### Current Limitations
- **Omitted Temporal Components in Baseline:** `hour`, `day`, `month`, and `day_of_week` were extracted in feature engineering but omitted from the baseline `ColumnTransformer` to maintain strict reproduction of experimental notebook dimensions.
- **Temporal Shift in Validation:** The baseline validation split was randomly stratified across 2019–2020 data, causing validation metrics to be more optimistic than out-of-time test metrics.
- **Synthetic Fractional Categoricals:** Applying standard Euclidean SMOTE over 1,254 one-hot encoded binary columns produces continuous fractional values (e.g., 0.35) that do not represent valid categorical states.
- **Probability Prior Shift:** Because SMOTE artificially inflated the training fraud prevalence to 10%, raw probability outputs from `predict_proba()` are uncalibrated relative to the natural 0.38% test base rate (mean predicted probability is ~4.85%).

### Planned Future Improvements
- **Chronological Validation Strategy:** Adopt the implemented `train_chronological_pipeline()` as the standard evaluation framework to eliminate temporal leakage.
- **Temporal Feature Inclusion:** Activate `create_improved_preprocessor()` to include cyclical sine/cosine encodings for transaction hour and day of week.
- **Alternative Resampling:** Benchmark `SMOTE-NC` (handling categorical data directly) or categorical frequency encoding to eliminate high-dimensional one-hot expansion.
- **Probability Calibration:** Apply isotonic regression or Platt scaling on out-of-fold validation predictions to produce calibrated probability outputs.
- **Cost-Sensitive Optimization:** Calibrate decision thresholds using a business cost matrix balancing undetected fraud loss against manual review costs.

---

## Author

- **Author:** Rahul
- **Repository:** [Humanli-Assignment](https://github.com/rahul15-manch/Humanli-Assignment)

---

## License

No license has been specified yet for this repository. All rights reserved.
