# Customer Churn Prediction & Explainable Machine Learning

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn and explains the factors influencing each prediction using SHAP.

The project covers the complete ML workflow — data cleaning, exploratory analysis, preprocessing, model training, evaluation, explainability, model serialization, and deployment through an interactive Streamlit dashboard.

## 🚀 Live Demo

**Streamlit App:** https://customer-churn-prediction-oxrqhanxcj2civoz4nd9ar.streamlit.app/

> The application allows users to enter customer information, receive a churn prediction and probability, and inspect the SHAP-based explanation behind the prediction.

---

## 🎯 Project Overview

Customer churn is a common business problem in subscription-based industries.

The objective of this project is to build a supervised machine learning system that can:

- Predict whether a customer is likely to churn
- Estimate the probability of churn
- Compare multiple machine learning approaches
- Evaluate classification performance using multiple metrics
- Explain individual predictions using SHAP
- Provide an interactive interface for real-time predictions
- Deploy the trained model as a web application

### Problem Type

**Supervised Learning → Binary Classification**

**Target variable:**

- `0` → Customer stays
- `1` → Customer churns

The probability output represents the model's estimated likelihood of churn. It should be interpreted as a predictive model output rather than a causal explanation.

---

## 📂 Dataset

This project uses the **IBM Telco Customer Churn** dataset.

The dataset contains approximately 7,000 telecom customers and includes demographic, account, service, contract, payment, and billing information.

### Important Features

| Category | Examples |
|---|---|
| Customer information | Gender, Senior Citizen, Partner, Dependents |
| Account information | Tenure, Contract, Paperless Billing |
| Services | Phone Service, Internet Service, Online Security |
| Support | Online Backup, Device Protection, Tech Support |
| Streaming | Streaming TV, Streaming Movies |
| Billing | Monthly Charges, Total Charges |
| Payment | Payment Method |
| Target | Churn |

### Data Cleaning

The dataset required several preprocessing steps:

1. Converted `TotalCharges` from string/object to numeric
2. Invalid numeric values were converted to missing values
3. Rows with missing `TotalCharges` were removed
4. `customerID` was removed because it is an identifier rather than a predictive feature
5. `Churn` was converted from `Yes/No` into `1/0`

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing Pipeline
     │
     ├── Numerical Features
     │      ├── Median Imputation
     │      └── StandardScaler
     │
     └── Categorical Features
            ├── Most-Frequent Imputation
            └── One-Hot Encoding
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── K-Nearest Neighbors
     ├── Decision Tree
     ├── Random Forest
     └── XGBoost
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1 Score
     ├── ROC-AUC
     └── Confusion Matrix
     │
     ▼
SHAP Explainability
     │
     ▼
Serialized ML Pipeline
     │
     ▼
Streamlit Deployment🤖 Models

The project explores multiple machine learning algorithms to demonstrate different approaches to binary classification.

Logistic Regression

Used as an interpretable baseline model.

It provides probability estimates and offers a useful benchmark for comparing more complex algorithms.

K-Nearest Neighbors

A distance-based algorithm that demonstrates the importance of feature scaling.

Decision Tree

A tree-based model capable of learning nonlinear relationships and feature interactions.

Random Forest

An ensemble of decision trees designed to improve robustness and predictive performance.

The deployed application uses the tuned Random Forest model.

XGBoost

A gradient boosting algorithm included to evaluate another strong approach for structured/tabular dataEvaluation Strategy

The model is evaluated using multiple metrics rather than relying on accuracy alone.

Accuracy — overall proportion of correct predictions
Precision — proportion of predicted churners who actually churned
Recall — proportion of actual churners detected by the model
F1 Score — balance between precision and recall
ROC-AUC — ability of the model to distinguish churners from non-churners
Confusion Matrix — detailed breakdown of correct and incorrect classifications

Because churn prediction can involve class imbalance and different business costs for false positives and false negatives, looking at multiple metrics provides a more complete evaluation.Model Explainability with SHAP

A major component of this project is Explainable AI.

Instead of only returning:

"This customer is predicted to churn."

the application also attempts to answer:

"Which features influenced this prediction?"

SHAP values are used to explain individual Random Forest predictions.

Interpretation
Positive SHAP values push the prediction toward churn
Negative SHAP values push the prediction toward no churn

The application displays the most influential features for each individual customer prediction.

SHAP explains the behavior of the trained model. It does not prove that a particular feature causes customer churn.🖥️ Streamlit Dashboard

The application provides an interactive interface containing:

Model Evaluation
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
ROC Curve
Customer Prediction

Users can enter:

Demographic information
Tenure
Contract type
Internet service
Support services
Streaming services
Payment method
Monthly charges
Total charges

The application then returns:

Churn / no-churn prediction
Estimated churn probability
SHAP-based explanation📁 Project Structure
customer-churn-prediction/
│
├── app.py
│   └── Streamlit dashboard and prediction interface
│
├── customer_churn_model.pkl
│   └── Serialized preprocessing + Random Forest pipeline
│
├── train_model.py
│   └── Reproducible model training script
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Excludes the local virtual environment
│
├── screenshots/
│   └── Application screenshots
│
└── README.md
    └── Project documentation
