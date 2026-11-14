import joblib
import pandas as pd


LABEL_MAP = {
    0: "Legitimate Transaction",
    1: "Fraudulent Transaction"
}

MANDATORY_COLUMNS = [
    "trans_date_trans_time",
    "merchant",
    "category",
    "amt",
    "gender",
    "state",
    "city_pop",
    "job",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long"
]


def validate_input_data(data):
    missing_cols = [col for col in MANDATORY_COLUMNS if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    null_cols = [col for col in MANDATORY_COLUMNS if data[col].isnull().any()]
    if null_cols:
        raise ValueError(f"Null values found in required columns: {sorted(null_cols)}")

    try:
        amt = pd.to_numeric(data["amt"])
        if (amt < 0).any():
            raise ValueError("Transaction amount (amt) must be non-negative.")
    except Exception as e:
        if isinstance(e, ValueError) and "non-negative" in str(e):
            raise
        raise ValueError("Transaction amount (amt) must be a valid numeric value.")

    try:
        city_pop = pd.to_numeric(data["city_pop"])
        if (city_pop < 0).any():
            raise ValueError("City population (city_pop) must be non-negative.")
    except Exception as e:
        if isinstance(e, ValueError) and "non-negative" in str(e):
            raise
        raise ValueError("City population (city_pop) must be a valid numeric value.")

    for lat_col in ["lat", "merch_lat"]:
        try:
            val = pd.to_numeric(data[lat_col])
            if (val < -90.0).any() or (val > 90.0).any():
                raise ValueError(f"{lat_col} must be between -90.0 and 90.0 degrees.")
        except Exception as e:
            if isinstance(e, ValueError) and "between" in str(e):
                raise
            raise ValueError(f"{lat_col} must be a valid numeric value.")

    for lon_col in ["long", "merch_long"]:
        try:
            val = pd.to_numeric(data[lon_col])
            if (val < -180.0).any() or (val > 180.0).any():
                raise ValueError(f"{lon_col} must be between -180.0 and 180.0 degrees.")
        except Exception as e:
            if isinstance(e, ValueError) and "between" in str(e):
                raise
            raise ValueError(f"{lon_col} must be a valid numeric value.")

    try:
        trans_time = pd.to_datetime(data["trans_date_trans_time"])
    except Exception:
        raise ValueError("trans_date_trans_time must be a valid datetime format.")

    try:
        dob = pd.to_datetime(data["dob"])
    except Exception:
        raise ValueError("dob must be a valid date format.")

    if (dob > trans_time).any():
        raise ValueError("Date of birth (dob) cannot be after transaction date.")


def predict_transaction(transaction, model_path="models/best_model.pkl", threshold=0.5):
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be a float between 0.0 and 1.0.")

    if isinstance(transaction, dict):
        data = pd.DataFrame([transaction])
    elif isinstance(transaction, pd.Series):
        data = pd.DataFrame([transaction.to_dict()])
    elif isinstance(transaction, pd.DataFrame):
        data = transaction.copy()
    else:
        raise ValueError("Transaction must be a dict, pandas Series, or pandas DataFrame.")

    if "is_fraud" in data.columns:
        data = data.drop(columns=["is_fraud"])

    validate_input_data(data)

    pipeline = joblib.load(model_path)
    probabilities = pipeline.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    labels = [LABEL_MAP[pred] for pred in predictions]

    if len(data) == 1:
        return labels[0], float(probabilities[0])

    result_df = pd.DataFrame({
        "predicted_class": labels,
        "fraud_probability": probabilities
    })
    return result_df


if __name__ == "__main__":
    sample_data = pd.read_csv("data/archive (5)/fraudTest.csv", nrows=5)

    print("Running sample predictions from fraudTest.csv:\n")
    for index, row in sample_data.iterrows():
        actual_label = LABEL_MAP.get(row.get("is_fraud"), "Unknown")
        predicted_class, fraud_prob = predict_transaction(row, threshold=0.5)
        print(f"Transaction {index + 1}:")
        print(f"  Merchant:          {row.get('merchant')}")
        print(f"  Category:          {row.get('category')}")
        print(f"  Amount:            ${row.get('amt'):.2f}")
        print(f"  Actual:            {actual_label}")
        print(f"  Predicted:         {predicted_class}")
        print(f"  Fraud Probability: {fraud_prob:.4f}\n")
