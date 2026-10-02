from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ---------------------------------------------------------
# Model path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_bundle.joblib"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


bundle = load_model()

model = bundle["model"]
preprocessor = bundle["preprocessor"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]
input_columns = bundle["raw_input_columns"]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    """
    Machine-learning dashboard for identifying potentially
    fraudulent credit-card transactions.
    """
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        model_name
    )

with col2:
    st.metric(
        "Classification Threshold",
        f"{threshold:.2f}"
    )

with col3:
    st.metric(
        "Input Features",
        len(input_columns)
    )

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------

st.divider()
st.header("📊 Model Performance")

st.write(
    """
    The final Random Forest model was evaluated on an
    untouched 15% test set. The classification threshold
    was selected using the validation set, not the test set.
    """
)

# Get saved final test metrics from the model bundle
final_metrics = bundle.get("final_test_metrics", {})

accuracy = final_metrics.get("Accuracy", 0.9995)
precision = final_metrics.get("Precision", 0.9455)
recall = final_metrics.get("Recall", 0.7324)
f1 = final_metrics.get("F1", 0.8254)
roc_auc = final_metrics.get("ROC-AUC", 0.9781)
pr_auc = final_metrics.get("PR-AUC", 0.8064)


# First row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Precision",
        f"{precision:.2%}",
        help="Of transactions predicted as fraud, how many were actually fraud."
    )

with col2:
    st.metric(
        "Recall",
        f"{recall:.2%}",
        help="Of all actual fraud transactions, how many were detected."
    )

with col3:
    st.metric(
        "F1 Score",
        f"{f1:.2%}",
        help="Balance between precision and recall."
    )


# Second row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.2%}"
    )

with col2:
    st.metric(
        "PR-AUC",
        f"{pr_auc:.2%}"
    )

with col3:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------

with st.expander("🧠 How was the model built?"):

    st.markdown(
        """
        ### Machine Learning Pipeline

        **1. Data Preparation**
        - Removed duplicate transactions
        - Checked missing values
        - Separated features and target variable

        **2. Train / Validation / Test Split**
        - 70% Training
        - 15% Validation
        - 15% Final Test

        **3. Feature Processing**
        - Standardized `Time` and `Amount`
        - Used anonymized PCA features `V1–V28`

        **4. Class Imbalance**
        - Applied SMOTE only to the training data
        - Validation and test data remained untouched

        **5. Models Compared**
        - Logistic Regression
        - Random Forest
        - XGBoost

        **6. Model Selection**
        - Models were compared on the validation set
        - PR-AUC was the primary selection metric
        - Random Forest achieved the strongest validation PR-AUC

        **7. Threshold Optimization**
        - Default threshold: `0.50`
        - Selected threshold: `0.68`
        - Threshold selection used validation data only

        **8. Final Evaluation**
        - Final performance was measured once on the
          untouched test set
        """
    )

# ---------------------------------------------------------
# Transaction input
# ---------------------------------------------------------

st.divider()

st.header("Transaction Analysis")

st.write(
    """
    Enter the transaction features below.

    `V1`–`V28` are anonymized PCA features from the original
    credit-card fraud dataset.
    """
)


# Time and Amount
col1, col2 = st.columns(2)

with col1:
    time_value = st.number_input(
        "Time",
        value=0.0,
        step=1.0
    )

with col2:
    amount_value = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=1.0
    )


# PCA features
feature_values = {}

st.subheader("Anonymized Transaction Features")

feature_columns = st.columns(4)

for i in range(1, 29):

    feature_name = f"V{i}"

    with feature_columns[(i - 1) % 4]:

        feature_values[feature_name] = st.number_input(
            feature_name,
            value=0.0,
            format="%.6f",
            key=feature_name
        )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.divider()

if st.button(
    "Analyze Transaction",
    type="primary",
    use_container_width=True
):

    transaction = {
        "Time": time_value,
        **feature_values,
        "Amount": amount_value
    }

    transaction_df = pd.DataFrame(
        [transaction]
    )

    # Preserve training column order
    transaction_df = transaction_df[input_columns]

    # Preprocessing
    processed_transaction = preprocessor.transform(
        transaction_df
    )

    # Fraud probability
    fraud_probability = model.predict_proba(
        processed_transaction
    )[:, 1][0]

    # Apply validation-selected threshold
    prediction = int(
        fraud_probability >= threshold
    )


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2%}"
        )


    if prediction == 1:

        st.error(
            "⚠️ Potential Fraud Detected"
        )

        st.write(
            """
            The predicted fraud probability exceeds the
            model's selected classification threshold.
            """
        )

    else:

        st.success(
            "✅ Transaction Classified as Legitimate"
        )

        st.write(
            """
            The predicted fraud probability is below the
            selected classification threshold.
            """
        )

# ---------------------------------------------------------
# Batch CSV Prediction
# ---------------------------------------------------------

st.divider()
st.header("📁 Batch Fraud Detection")

st.write(
    """
    Upload a CSV file containing multiple transactions.

    The file must contain the same input features used during
    model training: `Time`, `V1`–`V28`, and `Amount`.
    """
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        # Check required columns
        missing_columns = [
            column
            for column in input_columns
            if column not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            # Keep only model input columns
            prediction_data = batch_df[input_columns].copy()

            # Apply preprocessing
            processed_data = preprocessor.transform(
                prediction_data
            )

            # Predict fraud probabilities
            fraud_probabilities = model.predict_proba(
                processed_data
            )[:, 1]

            # Apply optimized threshold
            predictions = (
                fraud_probabilities >= threshold
            ).astype(int)

            # Add results
            results_df = batch_df.copy()

            results_df["Fraud_Probability"] = (
                fraud_probabilities
            )

            results_df["Prediction"] = predictions

            results_df["Risk"] = results_df[
                "Prediction"
            ].map(
                {
                    0: "Legitimate",
                    1: "Potential Fraud"
                }
            )

            # ---------------------------------------------
            # Summary
            # ---------------------------------------------

            st.subheader("Prediction Summary")

            total_transactions = len(results_df)

            fraud_count = int(
                results_df["Prediction"].sum()
            )

            legitimate_count = (
                total_transactions - fraud_count
            )

            fraud_rate = (
                fraud_count / total_transactions * 100
                if total_transactions > 0
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Transactions",
                f"{total_transactions:,}"
            )

            col2.metric(
                "Legitimate",
                f"{legitimate_count:,}"
            )

            col3.metric(
                "Potential Fraud",
                f"{fraud_count:,}"
            )

            col4.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%"
            )

            # ---------------------------------------------
            # Results table
            # ---------------------------------------------

            st.subheader("Transaction Results")

            st.dataframe(
                results_df[
                    [
                        "Time",
                        "Amount",
                        "Fraud_Probability",
                        "Risk"
                    ]
                ],
                use_container_width=True
            )

            # ---------------------------------------------
            # -------------------------------------------------
            # Fraud Analytics Dashboard
            # -------------------------------------------------

            st.divider()
            st.header("📊 Fraud Analytics Dashboard")

            def assign_risk_level(probability):
                if probability >= threshold:
                    return "High Risk"
                elif probability >= 0.30:
                    return "Medium Risk"
                return "Low Risk"

            results_df["Risk_Level"] = results_df[
                "Fraud_Probability"
            ].apply(assign_risk_level)

            st.subheader("Transaction Risk Distribution")

            risk_counts = (
                results_df["Risk_Level"]
                .value_counts()
                .reindex(
                    ["Low Risk", "Medium Risk", "High Risk"],
                    fill_value=0
                )
            )
            st.bar_chart(risk_counts)

            st.caption(
                f"Low Risk: <30% | Medium Risk: 30%–{threshold:.0%} | "
                f"High Risk: ≥{threshold:.0%}. The 30% boundary is a "
                "dashboard display category, not a learned model threshold."
            )

            st.subheader("Fraud vs Legitimate Transactions")
            prediction_counts = results_df["Risk"].value_counts()
            st.bar_chart(prediction_counts)

            st.subheader("Fraud Probability Distribution")

            probability_edges = sorted(set([
                0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, float(threshold), 0.80, 0.90, 1.00
            ]))

            probability_bins = pd.cut(
                results_df["Fraud_Probability"],
                bins=probability_edges,
                include_lowest=True
            )

            probability_distribution = (
                probability_bins.value_counts().sort_index()
            )

            probability_chart = pd.DataFrame({
                "Transaction Count": probability_distribution.values
            })
            probability_chart.index = (
                probability_distribution.index.astype(str)
            )
            st.bar_chart(probability_chart)

            st.subheader("💰 Transaction Amount Analysis")

            amount_summary = (
                results_df
                .groupby("Risk_Level", observed=False)["Amount"]
                .agg(
                    Transaction_Count="count",
                    Average_Amount="mean",
                    Total_Amount="sum"
                )
                .round(2)
                .reindex(["Low Risk", "Medium Risk", "High Risk"])
                .fillna(0)
            )

            st.dataframe(amount_summary, use_container_width=True)
            st.write("Average Transaction Amount by Risk Level")
            st.bar_chart(amount_summary["Average_Amount"])

            st.subheader("🚨 Top 10 Highest-Risk Transactions")

            top_risk_transactions = (
                results_df
                .sort_values("Fraud_Probability", ascending=False)
                .head(10)
            )

            top_risk_display = top_risk_transactions[
                [
                    "Time",
                    "Amount",
                    "Fraud_Probability",
                    "Risk_Level",
                    "Risk"
                ]
            ].copy()

            top_risk_display["Fraud_Probability"] = (
                top_risk_display["Fraud_Probability"]
                .map(lambda x: f"{x:.2%}")
            )

            st.dataframe(
                top_risk_display,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Highest Fraud Probabilities")

            top_risk_chart = (
                top_risk_transactions[["Fraud_Probability"]]
                .reset_index(drop=True)
            )
            top_risk_chart.index = [
                f"Transaction {i + 1}"
                for i in range(len(top_risk_chart))
            ]
            st.bar_chart(top_risk_chart)

            # Fraud-only table
            # ---------------------------------------------

            fraud_transactions = results_df[
                results_df["Prediction"] == 1
            ]

            st.subheader("⚠️ Flagged Transactions")

            if len(fraud_transactions) > 0:

                st.dataframe(
                    fraud_transactions[
                        [
                            "Time",
                            "Amount",
                            "Fraud_Probability",
                            "Risk"
                        ]
                    ],
                    use_container_width=True
                )

            else:

                st.success(
                    "No potentially fraudulent "
                    "transactions detected."
                )

            # ---------------------------------------------
            # Download results
            # ---------------------------------------------

            csv_results = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_results,
                file_name="fraud_prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Unable to process the file: {error}"
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    """
    Portfolio demonstration only. Predictions should not be
    used as the sole basis for real financial decisions.
    """
)
