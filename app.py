import datetime
import io
import pandas as pd
import streamlit as st

from src.predict import MANDATORY_COLUMNS, predict_transaction

CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel"
]

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
]

st.set_page_config(
    page_title="CreditGuard - Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .fraud-badge {
        background-color: #ffebee;
        color: #c62828;
        padding: 12px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
    .legit-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 12px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ CreditGuard")
st.sidebar.caption("AI-Powered Transaction Risk Scoring")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Single Transaction Prediction", "Batch CSV Prediction", "Model Information"]
)

if menu == "Dashboard":
    st.title("🛡️ CreditGuard Dashboard")
    st.subheader("Credit Card Fraud Detection System Overview")

    col_status1, col_status2 = st.columns(2)
    with col_status1:
        st.info("**Model Architecture:** SGDClassifier with Log Loss + SMOTE Resampling")
    with col_status2:
        st.success("**Model Status:** Active Production Baseline (`models/best_model.pkl`)")

    st.markdown("---")
    st.markdown("### Historical Benchmark Metrics")
    st.caption("Historical evaluation metrics computed on 1,296,675 training and 555,719 test transactions. These reflect historical performance under fixed evaluation conditions, not live performance guarantees.")

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    with row1_c1:
        st.metric("Validation PR-AUC", "0.247240", "Baseline")
    with row1_c2:
        st.metric("Final Test PR-AUC", "0.154851", "-37.4% shift")
    with row1_c3:
        st.metric("Validation F1-Score", "0.443414")
    with row1_c4:
        st.metric("Final Test F1-Score", "0.338452")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        st.metric("Validation Recall", "54.16%")
    with row2_c2:
        st.metric("Final Test Recall", "48.81%")
    with row2_c3:
        st.metric("Validation Precision", "37.53%")
    with row2_c4:
        st.metric("Final Test Precision", "25.90%")

    st.markdown("---")
    st.markdown("### Evaluation Summary Comparison")
    eval_df = pd.DataFrame([
        {
            "Split": "Validation (20% Stratified)",
            "Samples": "259,335",
            "Actual Frauds": "1,501",
            "Accuracy": "99.21%",
            "Precision": "37.53%",
            "Recall": "54.16%",
            "F1 Score": "0.4434",
            "PR-AUC": "0.2472"
        },
        {
            "Split": "Out-of-Time Test (fraudTest.csv)",
            "Samples": "555,719",
            "Actual Frauds": "2,145",
            "Accuracy": "99.26%",
            "Precision": "25.90%",
            "Recall": "48.81%",
            "F1 Score": "0.3385",
            "PR-AUC": "0.1549"
        }
    ])
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

elif menu == "Single Transaction Prediction":
    st.title("🔍 Single Transaction Risk Scoring")
    st.markdown("Enter transaction details to assess fraudulent transaction likelihood.")

    with st.expander("ℹ️ Decision Threshold Configuration", expanded=False):
        single_threshold = st.slider(
            "Classification Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Default is 0.50. The validation-optimal threshold of approximately 0.3894 increases fraud recall."
        )

    with st.form("single_transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Transaction Details")
            trans_date = st.date_input("Transaction Date", value=datetime.date(2020, 6, 21))
            trans_time = st.time_input("Transaction Time", value=datetime.time(14, 30, 0))
            amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=78.50, step=1.0)
            category = st.selectbox("Spending Category", CATEGORIES, index=CATEGORIES.index("grocery_pos"))
            merchant = st.text_input("Merchant Name", value="fraud_Kirlin and Sons")

        with col2:
            st.markdown("##### Cardholder Profile")
            gender = st.radio("Cardholder Gender", ["M", "F"], horizontal=True)
            dob = st.date_input("Date of Birth", value=datetime.date(1982, 5, 14), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
            job = st.text_input("Occupation", value="Electrical engineer")
            state = st.selectbox("State", STATES, index=STATES.index("CA"))
            city_pop = st.number_input("City Population", min_value=1, max_value=50000000, value=35000, step=1000)

        with col3:
            st.markdown("##### Geospatial Coordinates")
            lat = st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, value=34.0522, format="%.4f")
            long = st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, value=-118.2437, format="%.4f")
            merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=34.1200, format="%.4f")
            merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-118.3000, format="%.4f")

        submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

    if submitted:
        trans_datetime_str = f"{trans_date} {trans_time.strftime('%H:%M:%S')}"
        dob_str = str(dob)

        payload = {
            "trans_date_trans_time": trans_datetime_str,
            "merchant": merchant.strip(),
            "category": category,
            "amt": float(amt),
            "gender": gender,
            "state": state,
            "city_pop": int(city_pop),
            "job": job.strip(),
            "dob": dob_str,
            "lat": float(lat),
            "long": float(long),
            "merch_lat": float(merch_lat),
            "merch_long": float(merch_long)
        }

        try:
            pred_class, fraud_prob = predict_transaction(payload, threshold=single_threshold)

            st.markdown("---")
            st.markdown("### Transaction Analysis Result")

            res_c1, res_c2, res_c3 = st.columns([2, 2, 2])
            with res_c1:
                st.markdown("**Decision Outcome:**")
                if pred_class == "Fraudulent Transaction":
                    st.markdown('<div class="fraud-badge">🚨 Fraudulent Transaction</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="legit-badge">✅ Legitimate Transaction</div>', unsafe_allow_html=True)

            with res_c2:
                st.markdown("**Model Fraud Probability:**")
                st.markdown(f"### {fraud_prob * 100:.2f}%")
                st.progress(min(max(fraud_prob, 0.0), 1.0))

            with res_c3:
                st.markdown("**Classification Parameters:**")
                st.write(f"Applied Threshold: **{single_threshold:.2f}**")
                if fraud_prob >= single_threshold:
                    st.caption("Probability meets or exceeds decision threshold.")
                else:
                    st.caption("Probability is below decision threshold.")

            with st.expander("Submitted Payload Inspector"):
                st.json(payload)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

elif menu == "Batch CSV Prediction":
    st.title("📂 Batch CSV Transaction Scoring")
    st.markdown("Upload a CSV file containing transactions to generate predictions in bulk.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Batch Scoring Controls")
        batch_threshold = st.slider(
            "Batch Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01
        )

    st.markdown("##### 1. Prepare and Upload Transactions")
    st.markdown("The CSV must include the 13 mandatory columns: `trans_date_trans_time`, `merchant`, `category`, `amt`, `gender`, `state`, `city_pop`, `job`, `dob`, `lat`, `long`, `merch_lat`, `merch_long`.")

    sample_template = pd.DataFrame([{
        "trans_date_trans_time": "2020-06-21 12:14:25",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "gender": "M",
        "state": "SC",
        "city_pop": 333497,
        "job": "Naval architect",
        "dob": "1968-03-19",
        "lat": 33.9863,
        "long": -81.2007,
        "merch_lat": 34.5000,
        "merch_long": -81.5000
    }])

    template_csv = sample_template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Sample Template CSV",
        data=template_csv,
        file_name="creditguard_sample_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file contains {len(input_df):,} transaction rows.")

            missing_cols = [c for c in MANDATORY_COLUMNS if c not in input_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            else:
                if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing transactions through pipeline..."):
                        results_df = predict_transaction(input_df, threshold=batch_threshold)

                        merged_df = input_df.copy()
                        merged_df["predicted_class"] = results_df["predicted_class"]
                        merged_df["fraud_probability"] = results_df["fraud_probability"]

                        total_count = len(merged_df)
                        fraud_count = int((merged_df["predicted_class"] == "Fraudulent Transaction").sum())
                        legit_count = total_count - fraud_count
                        fraud_pct = (fraud_count / total_count * 100) if total_count > 0 else 0
                        avg_score = float(merged_df["fraud_probability"].mean())

                        st.markdown("---")
                        st.markdown("### Batch Scoring Summary")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Records", f"{total_count:,}")
                        c2.metric("Legitimate", f"{legit_count:,}")
                        c3.metric("Flagged Fraud", f"{fraud_count:,}", f"{fraud_pct:.2f}%")
                        c4.metric("Avg Fraud Score", f"{avg_score * 100:.2f}%")

                        st.markdown("#### Score Distribution")
                        hist_data = pd.cut(merged_df["fraud_probability"], bins=10).value_counts().sort_index()
                        hist_df = pd.DataFrame({"Score Range": [str(b) for b in hist_data.index], "Count": hist_data.values})
                        st.bar_chart(hist_df.set_index("Score Range"), use_container_width=True)

                        if fraud_count > 0:
                            st.markdown("#### Flagged High-Risk Transactions")
                            fraud_records = merged_df[merged_df["predicted_class"] == "Fraudulent Transaction"]
                            display_cols = [c for c in ["trans_date_trans_time", "merchant", "category", "amt", "fraud_probability"] if c in fraud_records.columns]
                            st.dataframe(fraud_records[display_cols].head(50), use_container_width=True)

                        output_buffer = io.StringIO()
                        merged_df.to_csv(output_buffer, index=False)
                        st.download_button(
                            "Download Scored Transactions CSV",
                            data=output_buffer.getvalue().encode("utf-8"),
                            file_name="creditguard_scored_transactions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {str(e)}")

elif menu == "Model Information":
    st.title("ℹ️ Model Architecture & Technical Specifications")

    st.markdown("### Pipeline Structure")
    st.code("""
Raw Payload
    │
    ▼
1. Feature Engineering (src/feature_engineering.py)
   - Haversine Distance (customer to merchant in km)
   - Customer Age at Transaction (years)
   - Datetime Extraction (hour, day, month, day_of_week)
   - Redundant Identifier Removal (cc_num, name, street, lat/long)
    │
    ▼
2. Preprocessing & Encoding (src/preprocessing.py)
   - StandardScaler on 4 numerical features:
     ['amt', 'city_pop', 'age', 'customer_merchant_distance_km']
   - OneHotEncoder(handle_unknown='ignore') on 5 categorical features:
     ['merchant' (693), 'category' (14), 'gender' (2), 'state' (51), 'job' (494)]
   - Output Feature Count: 1,258 features
    │
    ▼
3. Training Imbalance Handling (Training Fold Only)
   - SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=42)
   - Bypassed automatically during inference
    │
    ▼
4. Classifier (SGDClassifier)
   - loss='log_loss', max_iter=1000, tol=0.001, random_state=42
    │
    ▼
Output: Predicted Class & Probability
    """, language="text")

    st.markdown("---")
    st.markdown("### Technical Specifications")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Core Specifications:**")
        st.markdown("- **Model File:** `models/best_model.pkl`")
        st.markdown("- **Resampling:** SMOTE (10% minority ratio on training set)")
        st.markdown("- **Optimization:** Mini-batch SGD with Log Loss")
        st.markdown("- **Output Dimension:** 1,258 features")
        st.markdown("- **Default Threshold:** 0.50 (Adjustable in prediction UI)")
    with col_t2:
        st.markdown("**Documented Limitations:**")
        st.markdown("- **Temporal Features:** `hour`, `day`, `month`, `day_of_week` omitted in baseline ColumnTransformer to preserve notebook parity.")
        st.markdown("- **Temporal Split:** Baseline validation split was random stratified, causing optimistic validation metrics.")
        st.markdown("- **SMOTE on One-Hot Features:** Synthesizes fractional dummy variables.")
        st.markdown("- **Probability Calibration:** Raw probabilities shifted by SMOTE oversampling.")

    st.markdown("---")
    st.markdown("### Baseline vs. Chronological Model Reference")
    comp_table = pd.DataFrame([
        {
            "Model Variant": "Active Baseline (best_model.pkl)",
            "Split Strategy": "80/20 Random Stratified",
            "Val PR-AUC": "0.247240",
            "Test PR-AUC": "0.154851",
            "Val F1": "0.443414",
            "Test F1": "0.338452"
        },
        {
            "Model Variant": "Chronological (chronological_model.pkl)",
            "Split Strategy": "80/20 Time Cutoff (2020-03-06)",
            "Val PR-AUC": "0.204157",
            "Test PR-AUC": "0.144529",
            "Val F1": "0.356494",
            "Test F1": "0.306429"
        }
    ])
    st.dataframe(comp_table, use_container_width=True, hide_index=True)
