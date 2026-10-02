import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #050816;
    color: #F8FAFC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 1.05rem;
    margin-bottom: 2.5rem;
}


/* Cards */

.dashboard-card {
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.95),
        rgba(30, 41, 59, 0.75)
    );
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 18px;
    padding: 1.5rem;
    min-height: 125px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.card-label {
    color: #94A3B8;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.card-value {
    color: #F8FAFC;
    font-size: 1.55rem;
    font-weight: 700;
}


/* Metric cards */

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.95),
        rgba(30, 41, 59, 0.75)
    );
    border: 1px solid rgba(167, 139, 250, 0.18);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
}

.metric-label {
    color: #94A3B8;
    font-size: 0.85rem;
}

.metric-value {
    color: #F8FAFC;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 0.3rem;
}


/* Prediction */

.prediction-card {
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.98),
        rgba(30, 41, 59, 0.85)
    );
    border: 1px solid rgba(96, 165, 250, 0.25);
}

.prediction-title {
    font-size: 1rem;
    color: #94A3B8;
}

.prediction-result {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

.prediction-probability {
    font-size: 1.15rem;
    color: #CBD5E1;
}


/* Section headers */

.section-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    color: #F8FAFC;
}


/* Streamlit widgets */

div[data-baseweb="select"] > div {
    background-color: #0F172A;
    border-color: #334155;
}

div[data-baseweb="input"] > div {
    background-color: #0F172A;
    border-color: #334155;
}

input {
    color: #F8FAFC !important;
}

label {
    color: #CBD5E1 !important;
}


/* Button */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #2563EB, #7C3AED);
}

.stButton > button:hover {
    border: none;
    color: white;
    background: linear-gradient(90deg, #1D4ED8, #6D28D9);
}


/* Divider */

hr {
    border-color: rgba(148, 163, 184, 0.15);
}


/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    url = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/master/"
        "data/Telco-Customer-Churn.csv"
    )

    data = pd.read_csv(url)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna(subset=["TotalCharges"])

    data["Churn"] = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return data


df = load_dataset()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero-title">
    📊 Customer Churn Prediction
</div>

<div class="hero-subtitle">
    Machine Learning dashboard with model evaluation,
    customer-level predictions, and explainable AI
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROJECT OVERVIEW CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-label">MODEL</div>
<div class="card-value">Random Forest</div>
</div>
""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-label">TREES</div>
<div class="card-value">300</div>
</div>
""",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-label">PROBLEM TYPE</div>
<div class="card-value">Binary Classification</div>
</div>
""",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-label">CUSTOMERS</div>
<div class="card-value">7,000+</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Model Evaluation</div>',
    unsafe_allow_html=True
)


# Prepare evaluation data

evaluation_df = df.drop(columns=["customerID"])

X = evaluation_df.drop(columns=["Churn"])
y = evaluation_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Predictions

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


# Display metrics

metric_cols = st.columns(5)

metrics = [
    ("Accuracy", accuracy),
    ("Precision", precision),
    ("Recall", recall),
    ("F1 Score", f1),
    ("ROC-AUC", roc_auc)
]

for col, (label, value) in zip(metric_cols, metrics):

    with col:
        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">{label}</div>
<div class="metric-value">{value:.3f}</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# CONFUSION MATRIX + ROC CURVE
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Classification Performance</div>',
    unsafe_allow_html=True
)

chart_col1, chart_col2 = st.columns(2)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

with chart_col1:

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    # --------------------------------------------------------
    # FIX:
    # Dynamically choose black/white text depending on
    # background intensity.
    # --------------------------------------------------------

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            text_color = "white" if cm[i, j] > threshold else "black"

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color
            )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

with chart_col2:

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "ROC Curve",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter customer information below to estimate the probability of churn."
)


col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=monthly_charges * tenure
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🚀 Predict Customer Churn"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Prediction

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(
        customer_data
    )[0, 1]


    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        result_text = "⚠️ Customer is predicted to churn"
        result_color = "#F87171"

    else:

        result_text = "✅ Customer is predicted to stay"
        result_color = "#4ADE80"


    st.markdown(
        f"""
<div class="prediction-card">

<div class="prediction-title">
Prediction Result
</div>

<div class="prediction-result"
style="color:{result_color};">
{result_text}
</div>

<div class="prediction-probability">
Estimated churn probability:
<strong>{probability:.1%}</strong>
</div>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Why did the model make this prediction?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "SHAP shows which features pushed this prediction toward "
        "churn or toward no churn."
    )


    try:

        # Get preprocessing pipeline

        preprocessor = model.named_steps["preprocessor"]

        classifier = model.named_steps["model"]


        # Transform customer data

        transformed_customer = preprocessor.transform(
            customer_data
        )


        # Get feature names

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # Create SHAP explainer

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = explainer.shap_values(
            transformed_customer
        )


        # ----------------------------------------------------
        # Handle different SHAP versions
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            shap_row = shap_values[1][0]

        elif len(shap_values.shape) == 3:

            shap_row = shap_values[0, :, 1]

        else:

            shap_row = shap_values[0]


        # ----------------------------------------------------
        # CLEAN FEATURE NAMES
        # ----------------------------------------------------

        def clean_feature_name(name):

            name = str(name)

            # Remove pipeline prefixes
            name = re.sub(
                r"^(num|cat)__",
                "",
                name
            )

            # Convert OneHotEncoder names
            # Example:
            # InternetService_Fiber optic
            # -> Fiber optic internet

            if name.startswith("InternetService_"):

                value = name.replace(
                    "InternetService_",
                    ""
                )

                if value == "Fiber optic":
                    return "Fiber optic internet"

                if value == "DSL":
                    return "DSL internet"

                if value == "No":
                    return "No internet service"

                return f"{value} internet"


            if name.startswith("Contract_"):

                value = name.replace(
                    "Contract_",
                    ""
                )

                return f"{value} contract"


            if name.startswith("PaymentMethod_"):

                value = name.replace(
                    "PaymentMethod_",
                    ""
                )

                return value


            if "_" in name:

                parts = name.split("_", 1)

                feature = parts[0]
                value = parts[1]

                # Human-readable mappings

                readable_features = {
                    "gender": "Gender",
                    "Partner": "Partner",
                    "Dependents": "Dependents",
                    "PhoneService": "Phone service",
                    "MultipleLines": "Multiple lines",
                    "OnlineSecurity": "Online security",
                    "OnlineBackup": "Online backup",
                    "DeviceProtection": "Device protection",
                    "TechSupport": "Tech support",
                    "StreamingTV": "Streaming TV",
                    "StreamingMovies": "Streaming movies",
                    "PaperlessBilling": "Paperless billing"
                }

                readable_feature = readable_features.get(
                    feature,
                    feature
                )

                return f"{readable_feature}: {value}"


            readable_names = {
                "SeniorCitizen": "Senior citizen",
                "tenure": "Tenure",
                "MonthlyCharges": "Monthly charges",
                "TotalCharges": "Total charges"
            }

            return readable_names.get(
                name,
                name
            )


        clean_names = [
            clean_feature_name(name)
            for name in feature_names
        ]


        # ----------------------------------------------------
        # CREATE SHAP DATAFRAME
        # ----------------------------------------------------

        shap_df = pd.DataFrame({
            "Feature": clean_names,
            "SHAP": shap_row
        })


        shap_df["Absolute_SHAP"] = (
            shap_df["SHAP"].abs()
        )


        # Top 10 features

        top_features = (
            shap_df
            .sort_values(
                "Absolute_SHAP",
                ascending=False
            )
            .head(10)
            .sort_values("SHAP")
        )


        # ----------------------------------------------------
        # SHAP CHART
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        bar_colors = [
            "#60A5FA" if value < 0 else "#F87171"
            for value in top_features["SHAP"]
        ]


        ax.barh(
            top_features["Feature"],
            top_features["SHAP"],
            color=bar_colors
        )


        ax.axvline(
            0,
            color="black",
            linewidth=1
        )


        ax.set_title(
            "Top Features Influencing This Prediction",
            fontsize=14,
            fontweight="bold"
        )


        ax.set_xlabel(
            "SHAP Value"
        )


        ax.grid(
            axis="x",
            alpha=0.2
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # Explanation

        st.info(
            "🔴 Positive SHAP values push the prediction toward "
            "churn. 🔵 Negative SHAP values push the prediction "
            "toward no churn. SHAP explains the model's behavior "
            "but does not prove that a feature causes churn."
        )


    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated for this prediction."
        )

        st.caption(
            f"Technical details: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    Built with Python • Scikit-learn • Random Forest • SHAP • Streamlit
</div>
""",
    unsafe_allow_html=True
)