# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

An end-to-end machine learning pipeline that predicts customer churn for a telecom company using LightGBM, achieving a ROC-AUC score of 0.83.

---

## 📌 Problem Statement
Customer churn is one of the biggest challenges in the telecom industry. Retaining existing customers is significantly cheaper than acquiring new ones. This project builds a classifier to identify at-risk customers before they churn, enabling proactive retention strategies.

## 📊 Dataset
- Source: Telco Customer Churn Dataset (IBM Sample Data)
- Records: 7,043 customers
- Features: Contract type, tenure, monthly charges, internet service, payment method, and more
- Target: Churn (Yes/No)

## 🔍 Approach
1. **Exploratory Data Analysis** — identified key churn drivers: contract type, tenure, and monthly charges
2. **Class Imbalance** — handled using SMOTE (Synthetic Minority Oversampling Technique)
3. **Feature Engineering** — used ColumnTransformer pipeline for scaling and one-hot encoding
4. **Modelling** — trained LightGBM classifier with full preprocessing pipeline
5. **Hyperparameter Tuning** — optimised using GridSearchCV
6. **Evaluation** — assessed using ROC-AUC, classification report, and confusion matrix

## 📈 Results
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.83 |
| Model | LightGBM Classifier |

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/Sh10bh/CUSTOMER-CHURN-PREDICTION.git
cd CUSTOMER-CHURN-PREDICTION

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Customer_Churn_Prediction.ipynb
```

## 🛠 Tech Stack
- **Language:** Python
- **ML:** LightGBM, Scikit-learn, Imbalanced-learn (SMOTE)
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
