import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


def load_validation_data(val_path="data/processed/val_data.csv", train_path="data/archive (5)/fraudTrain.csv"):
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        X_val = val_df.drop(columns=["is_fraud"])
        y_val = val_df["is_fraud"]
        return X_val, y_val

    train_df = pd.read_csv(train_path)
    X = train_df.drop(columns=["is_fraud"])
    y = train_df["is_fraud"]
    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_val, y_val


def load_test_data(test_path="data/archive (5)/fraudTest.csv"):
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["is_fraud"])
    y_test = test_df["is_fraud"]
    return X_test, y_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob)
    }


def save_confusion_matrix_plot(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def evaluate_model(model_path="models/best_model.pkl"):
    pipeline = joblib.load(model_path)

    X_val, y_val = load_validation_data()
    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    print("=== VALIDATION EVALUATION ===")
    for metric_name, metric_val in val_metrics.items():
        print(f"{metric_name}: {metric_val:.6f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Legitimate", "Fraud"], zero_division=0))

    save_confusion_matrix_plot(
        y_val,
        y_val_pred,
        "Validation Confusion Matrix",
        "reports/confusion_matrix_val.png"
    )

    X_test, y_test = load_test_data()
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
    print("\n=== FINAL TEST EVALUATION ===")
    for metric_name, metric_val in test_metrics.items():
        print(f"{metric_name}: {metric_val:.6f}")
    print("\nFinal Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Legitimate", "Fraud"], zero_division=0))

    save_confusion_matrix_plot(
        y_test,
        y_test_pred,
        "Final Test Confusion Matrix",
        "reports/confusion_matrix_test.png"
    )

    summary_df = pd.DataFrame([
        {"Split": "Validation", **val_metrics},
        {"Split": "Test", **test_metrics}
    ])
    os.makedirs("reports", exist_ok=True)
    summary_df.to_csv("reports/evaluation_summary.csv", index=False)
    print("\nSaved evaluation summary to reports/evaluation_summary.csv")

    return val_metrics, test_metrics


if __name__ == "__main__":
    evaluate_model()
