import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Breast Cancer Classification Dashboard")
st.write("M.Tech AIML - Machine Learning Assignment 2")

st.markdown("---")

# Available trained models
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Upload test dataset
st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

# Model selection
st.subheader("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose an ML Model",
    list(model_options.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain the target column.")
    else:

        # Separate features and target
        X = data.drop(columns=["target"])
        y = data["target"]

        expected_features = list(scaler.feature_names_in_)

        if set(X.columns) != set(expected_features):
            st.error("Uploaded CSV features do not match training features.")
        elif not y.isin([0, 1]).all():
            st.error("Target must contain only 0 and 1.")
        else:

            # Maintain original feature order
            X = X[expected_features]

            # Scale uploaded data
            X_scaled = scaler.transform(X)

            # Load selected model
            model_path = os.path.join(
                "model",
                model_options[selected_model]
            )

            model = joblib.load(model_path)

            # Predictions
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            auc = roc_auc_score(y, probabilities)
            precision = precision_score(
                y, predictions, zero_division=0
            )
            recall = recall_score(
                y, predictions, zero_division=0
            )
            f1 = f1_score(
                y, predictions, zero_division=0
            )
            mcc = matthews_corrcoef(y, predictions)

            st.markdown("---")
            st.subheader("3. Model Evaluation Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")

            # Confusion matrix
            st.subheader("4. Confusion Matrix")

            cm = confusion_matrix(y, predictions)

            fig, ax = plt.subplots()

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"],
                ax=ax
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")

            st.pyplot(fig)
            plt.close(fig)

            # Classification report
            st.subheader("5. Classification Report")

            report = classification_report(
                y,
                predictions,
                target_names=["Malignant", "Benign"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

            # Prediction results
            st.subheader("6. Prediction Results")

            output = data.copy()
            output["Predicted Target"] = predictions
            output["Prediction Result"] = [
                "Benign" if p == 1 else "Malignant"
                for p in predictions
            ]

            st.dataframe(output)

            csv = output.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Prediction Results",
                csv,
                "prediction_results.csv",
                "text/csv"
            )

else:
    st.info("Please upload test_data.csv to evaluate your selected model.")

st.markdown("---")
st.caption("Machine Learning Assignment 2 | M.Tech AIML")
