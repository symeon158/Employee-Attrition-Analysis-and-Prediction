import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, f1_score)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
file_path = r""
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-7', sep=';')
except:
    df = pd.read_csv(file_path, encoding='cp1253', sep=';')

df.columns = df.columns.str.strip()

# --- 2. Rename Columns FIRST to avoid KeyErrors ---
df.rename(columns={
    'Αριθμός μητρώου': 'Registry Number', 
    'Φύλο': 'Gender', 
    'Ηλικία': 'Age',
    'Ημ/νία πρόσληψης': 'Hire Date',
    'Ημ/νία αποχώρησης': 'Departure Date',
    'Περιγραφή Αιτ. Αποχώρησης': 'Departure Reason Description', 
    'Ονομαστικός μισθός': 'Nominal Salary',
    'Σχέση Εργασίας': 'Work Relationship', 
    'Περιγραφή Υποκαταστήματος': 'City',
    'Διεύθυνση': 'Division', 
    'Ιδιότητα Προσωπικού': 'Job Property',
    'Θέση εργασίας': 'Job Position', 
    'GRADE': 'Grade', 
    'ΤΜΗΜΑ': 'Department'
}, inplace=True)

# --- 3. Feature Engineering & Filtering ---
# Convert date columns (using the new names)
date_cols = ['Hire Date', 'Departure Date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce', dayfirst=True)

today = datetime.today()
df['Tenure'] = (df['Departure Date'].fillna(today) - df['Hire Date']).dt.days // 365
df['Attrition'] = df['Departure Date'].notnull().astype(int)

# Filters
df = df[df['Work Relationship'] == 'ΑΟΡΙΣΤΟΥ ΧΡΟΝΟΥ']
df = df[(df['Departure Reason Description'] == 'VOLUNTARY DEPARTURE') | (df['Departure Reason Description'].isnull())]
df.loc[df['Department'].astype(str).str.contains('ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ', na=False), 'Attrition'] = 0
df['Gender'] = df['Gender'].replace({1: 'Male', 2: 'Female'})

# This line now works because 'Departure Date' exists!
df = df[(df['Departure Date'] > '2018-12-31') | (df['Departure Date'].isnull())]

# Salary & Grade Cleaning
df['Nominal Salary'] = df['Nominal Salary'].astype(str).str.replace(',', '.', regex=False)
df['Nominal Salary'] = pd.to_numeric(df['Nominal Salary'], errors='coerce')
df['Nominal Salary'] = df['Nominal Salary'].fillna(df['Nominal Salary'].median())
df['Job Property'] = df['Job Property'].fillna('OPERATIONAL')
df['Grade'] = df['Grade'].astype(str).replace({'99999': '0.9', '0,1': '0.99'}).astype(float)

# --- 4. Preprocessing for Machine Learning ---
# Drop non-numeric/raw date columns
cols_to_drop = ['Departure Date', 'Hire Date', 'Work Relationship', 'Departure Reason Description']
df_ml = df.drop(columns=cols_to_drop)

categorical_columns = ['Gender', 'City', 'Division', 'Job Property', 'Job Position', 'Department']
df_final = pd.get_dummies(df_ml, columns=categorical_columns, drop_first=True)

# Separate Target and Features
X = df_final.drop(columns=['Attrition', 'Registry Number'])
y = df_final['Attrition']

# Split Data (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Handle Class Imbalance
ratio = (y_train == 0).sum() / (y_train == 1).sum()

# --- 5. Model Training ---
xgb = XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
param_grid = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}

grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# --- 6. Evaluation ---
y_pred = best_model.predict(X_test)
print("\n--- Model Performance on Test Set ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Reds')
plt.title("Confusion Matrix (Unseen Data)")
plt.show()

# --- 7. Predicting on Active Employees ---
active_employees_df = df[df['Attrition'] == 0].copy()
# Align features with the training set
X_active_raw = df_final[df_final['Attrition'] == 0].drop(columns=['Attrition', 'Registry Number'])
X_active_scaled = scaler.transform(X_active_raw)

active_employees_df['Probability'] = best_model.predict_proba(X_active_scaled)[:, 1]
active_employees_df['Risk_Level'] = np.where(active_employees_df['Probability'] > 0.6, 'High', 
                                            np.where(active_employees_df['Probability'] > 0.3, 'Medium', 'Low'))

# --- 8. SHAP Explanation ---
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

print("\nGenerating SHAP Summary...")
shap.summary_plot(shap_values, X_train_raw)

# Final Output
print("\nTop 5 High-Risk Employees:")
print(active_employees_df[['Registry Number', 'Division', 'Probability']].sort_values(by='Probability', ascending=False).head())

# --- 9. Exporting Results for HR ---
# Select and reorder columns for a clean Excel report
hr_report = active_employees_df[[
    'Registry Number', 'Division', 'Department', 'Job Position', 
    'Tenure', 'Nominal Salary', 'Probability', 'Risk_Level'
]].copy()

# Sort by highest probability first
hr_report = hr_report.sort_values(by='Probability', ascending=False)

# Export to Excel
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
file_name = f"Attrition_Risk_Report_{timestamp}.xlsx"
hr_report.to_excel(file_name, index=False)
print(f"\nReport exported successfully: {file_name}")

# --- 10. Summary of Top Drivers (SHAP) ---
# This explains what features generally push people to leave
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_raw, show=False)
plt.title("Key Drivers of Employee Attrition")
plt.show()
