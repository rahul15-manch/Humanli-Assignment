import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from src.feature_engineering import prepare_features
from src.preprocessing import create_preprocessor


def build_pipeline():
    return Pipeline([
        ("feature_engineering", FunctionTransformer(prepare_features)),
        ("preprocessor", create_preprocessor()),
        ("smote", SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)),
        ("classifier", SGDClassifier(loss="log_loss", max_iter=1000, tol=0.001, random_state=42))
    ])


def evaluate_split(pipeline, X_val, y_val):
    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(y_val, y_val_pred, zero_division=0),
        "Recall": recall_score(y_val, y_val_pred, zero_division=0),
        "F1 Score": f1_score(y_val, y_val_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, y_val_prob),
        "PR-AUC": average_precision_score(y_val, y_val_prob)
    }

    print("Validation Results:")
    for metric_name, val in metrics.items():
        print(f"{metric_name:10s}: {val:.6f}")

    return metrics


def train_stratified_pipeline(data_path="data/archive (5)/fraudTrain.csv", model_output_path="models/best_model.pkl"):
    data = pd.read_csv(data_path)

    X = data.drop(columns=["is_fraud"])
    y = data["is_fraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_split(pipeline, X_val, y_val)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print("Saved trained pipeline to:", model_output_path)

    os.makedirs("data/processed", exist_ok=True)
    val_data = X_val.copy()
    val_data["is_fraud"] = y_val
    val_data.to_csv("data/processed/val_data.csv", index=False)

    return pipeline


def train_chronological_pipeline(data_path="data/archive (5)/fraudTrain.csv", model_output_path="models/chronological_model.pkl", train_ratio=0.8):
    data = pd.read_csv(data_path)

    parsed_time = pd.to_datetime(data["trans_date_trans_time"])
    sort_indices = parsed_time.argsort()
    data = data.iloc[sort_indices].reset_index(drop=True)

    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    val_data = data.iloc[split_index:]

    cutoff_time = train_data["trans_date_trans_time"].iloc[-1]
    val_start_time = val_data["trans_date_trans_time"].iloc[0]

    print("Chronological Split Details:")
    print(f"Training samples:   {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Training cutoff:    {cutoff_time}")
    print(f"Validation start:   {val_start_time}")

    X_train = train_data.drop(columns=["is_fraud"])
    y_train = train_data["is_fraud"]

    X_val = val_data.drop(columns=["is_fraud"])
    y_val = val_data["is_fraud"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate_split(pipeline, X_val, y_val)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print("Saved chronological pipeline to:", model_output_path)

    os.makedirs("data/processed", exist_ok=True)
    chronological_val = X_val.copy()
    chronological_val["is_fraud"] = y_val
    chronological_val.to_csv("data/processed/chronological_val_data.csv", index=False)

    return pipeline


def train_pipeline(data_path="data/archive (5)/fraudTrain.csv", model_output_path="models/best_model.pkl", split_strategy="stratified"):
    if split_strategy == "chronological":
        return train_chronological_pipeline(data_path=data_path, model_output_path=model_output_path)
    return train_stratified_pipeline(data_path=data_path, model_output_path=model_output_path)


if __name__ == "__main__":
    train_pipeline()
