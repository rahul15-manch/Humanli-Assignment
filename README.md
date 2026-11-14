# CreditGuard – Credit Card Fraud Detection Pipeline & Streamlit Application

CreditGuard is an end-to-end Machine Learning pipeline and interactive Streamlit web dashboard for credit card fraud detection on high-volume, highly imbalanced transaction streams.

The system encapsulates raw feature engineering, numerical scaling, categorical encoding, class imbalance handling via SMOTE, and an SGDClassifier optimized for stochastic log-loss minimization.

---

## Architecture Overview

```text
Raw Transaction Data (Dict, Series, or DataFrame)
    │
    ▼
1. Feature Engineering (src/feature_engineering.py)
   - Haversine customer-to-merchant distance calculation (km)
   - Customer age extraction from transaction timestamp and date of birth
   - Datetime component extraction (hour, day, month, day of week)
   - Redundant personal identifier pruning (card numbers, street, raw coordinates)
    │
    ▼
2. Preprocessing & Encoding (src/preprocessing.py)
   - StandardScaler on 4 numerical features:
     ['amt', 'city_pop', 'age', 'customer_merchant_distance_km']
   - OneHotEncoder(handle_unknown='ignore') on 5 categorical features:
     ['merchant', 'category', 'gender', 'state', 'job']
   - Total transformed feature dimensionality: 1,258 features
    │
    ▼
3. Class Imbalance Handling (Training Fold Only)
   - SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=42)
   - Resamples minority fraud class to 10% of majority legitimate class
   - Automatically bypassed during inference
    │
    ▼
4. Classification Engine
   - SGDClassifier(loss='log_loss', max_iter=1000, tol=0.001, random_state=42)
    │
    ▼
Inference Output: Predicted Class & Fraud Probability Score
```

---

## Directory Structure

```text
Machine Learning -Credit Card Fraud Detection/
├── app.py                          # Interactive Streamlit dashboard application
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── PROJECT_TECHNICAL_HANDOFF.md    # Detailed technical specification & audit handoff
├── data/
│   ├── archive (5)/
│   │   ├── fraudTrain.csv          # Training dataset (1,296,675 records)
│   │   └── fraudTest.csv           # Out-of-time test dataset (555,719 records)
│   └── processed/
│       ├── val_data.csv            # Stratified validation split
│       └── chronological_val_data.csv # Chronological validation split
├── models/
│   ├── best_model.pkl              # Production pipeline artifact (702 KB)
│   ├── chronological_model.pkl     # Experimental chronological split model artifact
│   ├── tuned_sgd_model.pkl         # Experimental tuned SGD weights
│   └── tuned_sgd_parameters.pkl    # Pickled tuned hyperparameters
├── notebooks/                      # Exploratory research notebooks
│   ├── eda.ipynb
│   ├── data_preprocessing.ipynb
│   ├── imbalance_handling.ipynb
│   ├── model_training.ipynb
│   └── model_tuning.ipynb
├── reports/                        # Evaluation metrics and confusion matrices
│   ├── confusion_matrix_val.png
│   ├── confusion_matrix_test.png
│   └── evaluation_summary.csv
└── src/                            # Modular production source code
    ├── __init__.py
    ├── feature_engineering.py      # Feature engineering and Haversine distance
    ├── preprocessing.py            # ColumnTransformer definitions
    ├── train.py                    # Pipeline assembly and training workflows
    ├── evaluate.py                 # Evaluation reports and confusion matrix plots
    └── predict.py                  # Validated single and batch inference endpoint
```

---

## Environment Setup & Installation

### Prerequisites
- Python 3.10+
- macOS, Linux, or Windows with virtual environment support

### Setup Steps
```bash
# 1. Clone repository and navigate to root directory
cd "Machine Learning -Credit Card Fraud Detection "

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## Model Artifact Locations

- **Production Baseline Model:** `models/best_model.pkl`
  - Encapsulates feature engineering, column transformation, SMOTE bypass, and classifier weights.
  - Ingests raw dictionaries, pandas Series, and DataFrames directly.
- **Chronological Split Experimental Model:** `models/chronological_model.pkl`

---

## Execution Commands

### 1. Launch Streamlit Application
```bash
streamlit run app.py
```
Access the dashboard in your web browser at `http://localhost:8501`.

### 2. Run Single or Sample Prediction
```bash
python -m src.predict
```

### 3. Evaluate Pipeline on Validation and Test Datasets
```bash
python -m src.evaluate
```

### 4. Train Model Pipeline
```bash
# Train baseline stratified pipeline (saves to models/best_model.pkl)
python -m src.train

# Or train chronological model via Python API:
# from src.train import train_chronological_pipeline
# train_chronological_pipeline()
```

---

## Streamlit Application Workflows

### 1. Dashboard
- Displays active model status and verified benchmark evaluation metrics.
- Presents side-by-side performance comparison across validation and out-of-time test sets.

### 2. Single Transaction Prediction
- Provides an input form for all 13 mandatory transaction fields.
- Allows real-time classification threshold adjustments (default `0.50`, recommended `0.3894`).
- Outputs an actionable visual badge (`Legitimate Transaction` vs. `Fraudulent Transaction`) with model fraud probability and payload inspector.

### 3. Batch CSV Prediction
- Allows dragging and dropping CSV transaction batches.
- Validates uploaded CSV schema against mandatory input fields.
- Displays summary statistics (total records, flagged frauds, fraud percentage, average score).
- Renders an interactive score distribution bar chart and high-risk transaction review table.
- Generates a downloadable CSV with original columns, predicted class labels, and fraud probabilities.

### 4. Model Information
- Details the end-to-end architecture, 1,258-feature space breakdown, resampling rationale, and known limitations.

---

## Verified Baseline Model Metrics

| Metric | Validation Dataset (`data/processed/val_data.csv`) | Final Out-of-Time Test Set (`data/archive (5)/fraudTest.csv`) |
|:---|:---:|:---:|
| **Sample Count** | 259,335 | 555,719 |
| **Actual Fraud Records** | 1,501 (0.578%) | 2,145 (0.386%) |
| **Accuracy** | 99.2130% | 99.2635% |
| **Precision** | 37.5346% | 25.9030% |
| **Recall** | 54.1639% | 48.8112% |
| **F1-Score** | 0.443414 | 0.338452 |
| **ROC-AUC** | 0.897223 | 0.858155 |
| **PR-AUC (Average Precision)** | **0.247240** | **0.154851** |

### Verified Confusion Matrix
- **Validation:** True Negatives: `256,481` | False Positives: `1,353` | False Negatives: `688` | True Positives: `813`
- **Final Test:** True Negatives: `550,579` | False Positives: `2,995` | False Negatives: `1,098` | True Positives: `1,047`

---

## Known Model Limitations & Architectural Notes

1. **Temporal Disconnect in Validation Split:** The raw data partitioned `fraudTrain.csv` and `fraudTest.csv` chronologically (2019 to mid-2020 vs. mid-2020 to end-2020). The original baseline validation split was random stratified, producing higher validation metrics (PR-AUC 0.2472) than out-of-time test metrics (PR-AUC 0.1548). Use `train_chronological_pipeline()` to avoid this in future model iterations.
2. **Omitted Datetime Components:** `hour`, `day`, `month`, and `day_of_week` were extracted in feature engineering but omitted from the baseline `ColumnTransformer` to preserve notebook reproduction. They can be enabled in future models using `create_improved_preprocessor()`.
3. **SMOTE on One-Hot Encodings:** Standard SMOTE creates fractional interpolations over binary dummy columns. Consider SMOTE-NC or frequency encoding for future iterations.
4. **Probability Calibration:** Raw predicted probabilities are shifted upward by the 10% SMOTE training prior (mean predicted probability ~4.85% vs base rate 0.58%). Decision thresholds can be tuned downward (e.g., `0.3894`) to recover higher fraud recall.
