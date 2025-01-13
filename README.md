# Employee Attrition Analysis and Prediction

This repository showcases a comprehensive project dedicated to analyzing and predicting employee attrition using advanced machine learning techniques. The project also offers insightful visualizations to help organizations identify and mitigate employee turnover risks effectively.

---

## 🔍 Project Overview

Employee attrition, or turnover, is a critical challenge that impacts organizational stability and growth. This project aims to:
- **Analyze** historical attrition data to uncover key trends and patterns.
- **Predict** future attrition using robust machine learning models, including XGBoost.
- **Visualize** turnover insights across various dimensions, such as department, tenure, and demographics, to empower data-driven decision-making.

---

## 📊 Features

### 🛠 Data Preprocessing
- Comprehensive data preparation, including:
  - Handling missing values and outliers.
  - Encoding categorical variables.
  - Feature engineering (e.g., creating `Tenure`, `Attrition Flag`, and `Age` variables).
  - Scaling and balancing datasets to address class imbalance.

### 🤖 Advanced Machine Learning
- Built and evaluated multiple machine learning models, such as:
  - **XGBoost** (achieved the highest ROC-AUC score of 0.885).
  - Random Forest, Logistic Regression, and Gradient Boosting.
- Fine-tuned models with **GridSearchCV** for optimal performance.
- Applied 5-fold cross-validation to ensure model robustness and generalizability.

### 📈 Insightful Visualizations
- Dynamic visualizations showcasing attrition trends over:
  - Time periods.
  - Departments, genders, and tenure groups.
- Annotated charts highlighting year-over-year differences for actionable insights.

### 🧠 Explainability and Interpretability
- Leveraged **SHAP (SHapley Additive exPlanations)** to:
  - Identify top predictors of employee attrition.
  - Provide division-level insights to help HR proactively address retention risks.

---

## 🚀 Key Outcomes
- **Predictive Power**: Achieved a high-performing XGBoost model with an ROC-AUC of 0.885.
- **Strategic Insights**: Delivered division-level attrition insights, enabling HR teams to implement targeted retention strategies.
- **Data-Driven Decision-Making**: Created a pipeline of visual and statistical tools to empower HR management.
![image](https://github.com/user-attachments/assets/769ee17f-7b77-49fc-98ca-72492544645c)

---

## 🛠 Technologies Used
- **Programming & Data Handling**: Python (Pandas, NumPy, Scikit-learn)
- **Machine Learning**: XGBoost, Random Forest, Logistic Regression, Gradient Boosting
- **Model Optimization**: GridSearchCV
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Visualization**: Matplotlib, Seaborn

---

