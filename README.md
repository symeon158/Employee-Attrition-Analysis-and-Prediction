# 🚀 Enterprise Employee Attrition Prediction & Analytics

This repository showcases a comprehensive, end-to-end machine learning pipeline designed to analyze historical turnover and predict future employee attrition. By leveraging **XGBoost** and **SHAP (SHapley Additive exPlanations)**, this project provides a data-driven bridge between raw HR data and strategic retention planning.

---

## 🔍 Project Overview

Employee attrition is a critical challenge impacting organizational stability. This project moves beyond descriptive statistics to provide **prescriptive insights**:
- **Analyze**: Identify historical trends and demographic correlations.
- **Predict**: Quantify the probability of departure for currently active employees.
- **Explain**: Use Explainable AI (XAI) to understand the "why" behind every risk flag.

---

## 📊 Performance Benchmarks

Our refined XGBoost model demonstrates high reliability on unseen test data, ensuring that HR interventions are targeted and effective.
Based on unseen test data, our optimized XGBoost model achieved:
- **ROC-AUC Score**: **0.931** (Excellent separation power)
- **F1-Score (Attrition)**: **0.81**
- **Recall**: **0.80** (Correctly identifying 8 out of 10 flight risks)

<img width="1500" height="600" alt="matrix_roc" src="https://github.com/user-attachments/assets/87c08ff2-1ba7-4c55-8d60-a78a6dcb8085" />



---

## 🛠 Features

### 1. Robust Data Preprocessing
- **Localized Encoding**: Specialized handling for Greek character sets (`ISO-8859-7`).
- **Feature Engineering**: Dynamic calculation of `Tenure` and `Age` using temporal logic.
- **Cleaning**: Robust handling of null values and specialized "Departure Reason" filtering to focus exclusively on **Voluntary Departures**.

### 2. Advanced Machine Learning Pipeline
- **Algorithm**: Optimized **XGBoost Classifier**.
- **Class Imbalance**: Utilized `scale_pos_weight` to address the rarity of attrition events in healthy organizations.
- **Optimization**: **GridSearchCV** with 5-fold Stratified Cross-Validation to ensure the model generalizes to new hires.
- **Leakage Prevention**: Strict separation of training and testing data.

### 3. Explainability & Interpretability (XAI)
We utilize **SHAP** to decode the "black box" of machine learning:
- **Global Importance**: Rankings of which factors (Salary, Tenure, Grade) drive turnover company-wide.
- **Local Explanations**: Identification of specific risk factors for individual employees to assist in "Stay Interviews."

<img width="750" height="500" alt="Shap_Tenure_Grade" src="https://github.com/user-attachments/assets/7ac3f280-3554-4450-a485-5c6a641ca9e3" />



---

## 🚀 Key Strategic Outcomes

- **Risk Segmenting**: Automated categorization of employees into **High, Medium, and Low** risk tiers.
- **Operational Reporting**: Automated generation of prioritized Excel reports for HR Business Partners and Division Heads.
- **Targeted Retention**: Identification of high-risk departments and divisions allowing for localized cultural or structural interventions.



---

## 🛠 Technologies Used

- **Languages**: Python (Pandas, NumPy)
- **Machine Learning**: Scikit-Learn, XGBoost
- **Optimization**: GridSearchCV
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Visualization**: Matplotlib, Seaborn

---


