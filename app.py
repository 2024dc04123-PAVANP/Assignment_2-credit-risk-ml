import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
    roc_curve, classification_report
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# THEME TOGGLE
# ==================================================
theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"], horizontal=True)

if theme == "Dark":
    BG = "#000000"
    SIDEBAR = "#111111"
    CARD = "#1a1a1a"
    TEXT = "#ffffff"
    BORDER = "#333333"
else:
    BG = "#e8f1ff"
    SIDEBAR = "#d6e6ff"
    CARD = "#ffffff"
    TEXT = "#0b1e3d"
    BORDER = "#b3ccff"

# ==================================================
# CSS
# ==================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
        font-family: "Segoe UI", Arial, sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR};
        border-right: 2px solid {BORDER};
    }}

    section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}

    div[data-testid="stFileUploader"] * {{
        color: {TEXT} !important;
        background-color: {CARD} !important;
    }}

    .stMetric {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px;
    }}

    button[data-baseweb="tab"] {{
        color: {TEXT} !important;
        font-weight: 600;
    }}

    .stDownloadButton button {{
        background-color: #2563eb;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
    }}

    /* ================= SELECTBOX DARK MODE FIX ================= */
    div[data-baseweb="select"] > div {{
        background-color: {CARD} !important;
    }}

    div[data-baseweb="select"] span {{
        color: {TEXT} !important;
    }}

    ul[role="listbox"] li {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
    }}
    /* =========================================================== */
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;">💳 Credit Card Default Prediction</h1>
    <p style="text-align:center;">Machine Learning Model Evaluation Dashboard</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==================================================
# LOAD MODELS
# ==================================================
models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "KNN": joblib.load("model/knn_model.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
    "Random Forest": joblib.load("model/random_forest_model.pkl"),
    "XGBoost": joblib.load("model/xgboost_model.pkl"),
}

scaler = joblib.load("model/scaler.pkl")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown("## ⚙️ Configuration")
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# ==================================================
# MAIN LOGIC
# ==================================================
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if len(data.columns) == 1:
        data = data.iloc[:, 0].str.split(",", expand=True)
        data.columns = data.iloc[0]
        data = data[1:].reset_index(drop=True)

    data.columns = data.columns.astype(str).str.strip()

    if "ID" in data.columns:
        data.drop(columns=["ID"], inplace=True)

    target = "default.payment.next.month"
    if target not in data.columns:
        st.error("Target column not found.")
        st.stop()

    X = data.drop(columns=[target])
    y = data[target]

    X_proc = scaler.transform(X) if selected_model in ["Logistic Regression", "KNN"] else X
    model = models[selected_model]

    y_pred = model.predict(X_proc)
    y_prob = model.predict_proba(X_proc)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    tabs = st.tabs([
        "📌 Overview",
        "📊 Metrics",
        "📈 ROC Curve",
        "🧩 Confusion Matrix",
        "📋 Classification Report",
        "📊 Model Comparison",
        "📥 Download"
    ])

    # ---------------- Overview ----------------
    with tabs[0]:
        st.markdown("## 📌 Project Overview")

        st.markdown(
            """
            **Project Title:** Credit Card Default Prediction  
            **Problem Type:** Binary Classification  

            This project predicts whether a credit card customer will  
            **default on their next payment** using machine learning models.
            """
        )

        st.markdown("### 📊 Dataset Information")
        st.markdown(
            """
            - Credit Card Default Dataset (Kaggle / UCI)
            - Records: 30,000 customers
            - Target Variable: `default.payment.next.month`
            - 0 → No Default | 1 → Default
            """
        )

        st.markdown("### 🤖 Models Used")
        st.markdown(
            """
            - Logistic Regression  
            - Decision Tree  
            - KNN  
            - Naive Bayes  
            - Random Forest  
            - XGBoost
            """
        )

        st.markdown("### 📈 Evaluation Metrics")
        st.markdown("Accuracy, AUC, Precision, Recall, F1 Score, MCC")

        st.markdown("---")

        st.markdown(
            """
            **Student Name:** **PAVAN P**  
            **BITS ID:** **2024DC04123**
            """
        )

        st.success(f"Currently Selected Model: **{selected_model}**")

    # ---------------- Metrics ----------------
    with tabs[1]:
        c1 = st.columns(3)
        c1[0].metric("Accuracy", f"{acc:.3f}")
        c1[1].metric("AUC", f"{auc:.3f}")
        c1[2].metric("MCC", f"{mcc:.3f}")

        c2 = st.columns(3)
        c2[0].metric("Precision", f"{prec:.3f}")
        c2[1].metric("Recall", f"{rec:.3f}")
        c2[2].metric("F1 Score", f"{f1:.3f}")

    # ---------------- ROC Curve ----------------
    with tabs[2]:
        fpr, tpr, _ = roc_curve(y, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    # ---------------- Confusion Matrix ----------------
    with tabs[3]:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # ---------------- Classification Report ----------------
    with tabs[4]:
        report = classification_report(
            y, y_pred,
            target_names=["No Default", "Default"],
            output_dict=True
        )
        df = pd.DataFrame(report).transpose().round(3)
        st.table(df)

    # ---------------- Model Comparison ----------------
    with tabs[5]:
        rows = []
        for name, mdl in models.items():
            X_eval = scaler.transform(X) if name in ["Logistic Regression", "KNN"] else X
            yp = mdl.predict(X_eval)
            yp_prob = mdl.predict_proba(X_eval)[:, 1]

            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y, yp), 3),
                "AUC": round(roc_auc_score(y, yp_prob), 3),
                "Precision": round(precision_score(y, yp), 3),
                "Recall": round(recall_score(y, yp), 3),
                "F1": round(f1_score(y, yp), 3),
                "MCC": round(matthews_corrcoef(y, yp), 3),
            })

        comp_df = pd.DataFrame(rows)
        st.table(comp_df)

    # ---------------- Download ----------------
    with tabs[6]:
        out = data.copy()
        out["Prediction"] = y_pred
        out["Probability"] = y_prob
        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False),
            file_name="credit_default_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV file to begin.")
