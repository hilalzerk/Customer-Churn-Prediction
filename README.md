# Customer Churn Prediction

## Project Overview
This project aims to predict whether a telecom customer will churn or not, using machine learning models. 
The dataset is from Kaggle Playground Series S6E3.

## Dataset
- **Source:** [Kaggle - Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
- **Train set:** 594,194 rows
- **Features:** 20 features + 1 target (Churn)
- **Target:** Churn (1 = churned, 0 = stayed)
- **Metric:** AUC-ROC

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## Project Steps
1. Exploratory Data Analysis (EDA)
2. Feature Engineering (One-Hot Encoding)
3. Model Training & Evaluation
4. Kaggle Submission

## Model Results
| Model | AUC-ROC | Accuracy |
|---|---|---|
| Logistic Regression | 0.9137 | 0.86 |
| Random Forest | 0.8976 | 0.85 |
| **XGBoost** | **0.9143** | **0.86** |

## Kaggle Score
Public Score: **0.9058** (XGBoost)
