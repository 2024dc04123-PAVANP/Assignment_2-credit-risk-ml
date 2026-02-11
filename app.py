import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

st.title("💳 Credit Card Default Prediction App")
st.write("Machine Learning Classification Models Demo")

# ---------------------------
# Load models
# ---------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "KNN": joblib.load("model/knn_model.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
    "Random Forest": joblib.load("model/random_forest_model.pkl"),
    "XGBoost": joblib.load("model/xgboost_model.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Select a Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ---------------------------
# Main Logic
# ---------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Test Data")
    st.write(data.head())

    # Target column
    target_col = "default.payment.next.month"

    if target_col not in data.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
    else:
        X_test = data.drop(columns=[target_col])
        y_test = data[target_col]

        # Scale only if needed
        if selected_model_name in ["Logistic Regression", "KNN"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test

        model = models[selected_model_name]

        # Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("AUC", f"{auc:.3f}")
        col3.metric("MCC", f"{mcc:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{precision:.3f}")
        col5.metric("Recall", f"{recall:.3f}")
        col6.metric("F1 Score", f"{f1:.3f}")

        # Confusion Matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
