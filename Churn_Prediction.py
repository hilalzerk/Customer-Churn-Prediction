# =============================================================
# Project   : Customer Churn Prediction
# Dataset   : Kaggle Playground Series S6E3
# Goal      : Predict whether a customer will churn or not
# Target    : Churn (1 = churned, 0 = stayed)
# Metric    : AUC-ROC
# Author    : Hilal Zerk Demirkan
# Date      : March 2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("Projeler/Churn Prediction/Data/train.csv")
test = pd.read_csv("Projeler/Churn Prediction/Data/test.csv")
# Use a sample for faster processing during EDA
train_sample = train.sample(n=50000, random_state=42)

# Load the data
train_sample.shape
train_sample.head()
train_sample.info()
train_sample.describe()
# Check the distribution of the target variable (how many customers churned vs. stayed)
train_sample["Churn"].value_counts()
train_sample.isnull().sum()

# =============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the target variable
sns.countplot(x="Churn", data=train_sample)
plt.title("Churn Distribution")
plt.savefig("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/Plots/churn_distribution.png", bbox_inches="tight", dpi=150)
plt.show()


# Check data types
train_sample.dtypes
# Check the distribution of numerical columns
train_sample[["tenure", "MonthlyCharges", "TotalCharges"]].describe()

# Visualize the distribution of numerical columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.histplot(train_sample[col], ax=axes[i], kde=True)
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.savefig("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/Plots/numerical_distributions.png", bbox_inches="tight", dpi=150)
plt.show()


# Visualize numerical columns vs Churn
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.boxplot(x="Churn", y=col, data=train_sample, ax=axes[i])
    axes[i].set_title(f"{col} vs Churn")

plt.tight_layout()
plt.savefig("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/Plots/numerical_vs_churn.png", bbox_inches="tight", dpi=150)
plt.show()


# Visualize categorical columns vs Churn
cat_cols = ["gender", "Partner", "Dependents", "PhoneService",
            "InternetService", "Contract", "PaymentMethod"]

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, hue="Churn", data=train_sample, ax=axes[i])
    axes[i].set_title(f"{col} vs Churn")
    axes[i].tick_params(axis='x', rotation=15)
for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/Plots/categorical_vs_churn.png", bbox_inches="tight", dpi=72)
plt.show()

# =============================================================
# FEATURE ENGINEERING
# =============================================================

# Convert target variable to binary
train_sample["Churn"] = train_sample["Churn"].map({"Yes": 1, "No": 0})

# Verify the conversion
train_sample["Churn"].value_counts()


# Convert categorical columns to numerical using one-hot encoding
train_final = pd.get_dummies(train_sample, columns=[
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
])

# Separate features and target variable
X = train_final.drop(columns=["id", "Churn"])
y = train_final["Churn"]

# Verify the shapes
X.shape, y.shape


# Split the data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes
X_train.shape, X_test.shape

# X_train: 40,000
# X_test: 10,000

# Scale the data and retrain
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred = lr_model.predict(X_test_scaled)
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))

# Logistic Regression Results:
# AUC-ROC: 0.9137
# Accuracy: 0.86
# Class 0 (stayed)  - Precision: 0.91, Recall: 0.91, F1: 0.91
# Class 1 (churned) - Precision: 0.69, Recall: 0.68, F1: 0.68


# Train a Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf))

# Random Forest Results:
# AUC-ROC: 0.8976
# Accuracy: 0.85
# Class 0 (stayed)  - Precision: 0.89, Recall: 0.92, F1: 0.90
# Class 1 (churned) - Precision: 0.68, Recall: 0.60, F1: 0.64


# Train an XGBoost model
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="auc")
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_xgb))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_xgb))

# XGBoost Results:
# AUC-ROC: 0.9143
# Accuracy: 0.86
# Class 0 (stayed)  - Precision: 0.90, Recall: 0.92, F1: 0.91
# Class 1 (churned) - Precision: 0.70, Recall: 0.65, F1: 0.68


# Model Comparison:
# -------------------------------------------------------
# Model                  AUC-ROC    Accuracy
# -------------------------------------------------------
# Logistic Regression    0.9137     0.86
# Random Forest          0.8976     0.85
# XGBoost                0.9143     0.86  (best model)
# -------------------------------------------------------


# =============================================================
# SUBMISSION
# =============================================================

# Apply the same encoding to test data
test_final = pd.get_dummies(test, columns=[
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
])

# Align test data with training data
test_final = test_final.reindex(columns=X.columns, fill_value=0)

# Predict
y_prob_submission = xgb_model.predict_proba(test_final)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "Churn": y_prob_submission
})

submission.to_csv("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/submission.csv", index=False)

submission.head()


# =============================================================
# MODEL COMPARISON
# =============================================================

# Visualize model comparison
models = ["Logistic Regression", "Random Forest", "XGBoost"]
auc_scores = [0.9137, 0.8976, 0.9143]

sns.barplot(x=models, y=auc_scores)
plt.title("Model Comparison - AUC-ROC Scores")
plt.ylim(0.85, 0.92)
plt.ylabel("AUC-ROC")
plt.savefig("/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Projeler/Churn Prediction/Plots/model_comparison.png", bbox_inches="tight", dpi=72)
plt.show()