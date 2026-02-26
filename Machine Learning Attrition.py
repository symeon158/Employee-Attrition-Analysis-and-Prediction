import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Load & Clean Data ---
file_path = r"path"
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-7', sep=';')
except:
    df = pd.read_csv(file_path, encoding='cp1253', sep=';')

df.columns = df.columns.str.strip()

# Date conversions
date_columns = ['Ημ/νία γέννησης', 'Ημ/νία πρόσληψης', 'Ημ/νία αποχώρησης']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce', dayfirst=True)

# Feature Engineering
today = datetime.today()
if 'Ημ/νία πρόσληψης' in df.columns:
    df['Tenure'] = (df['Ημ/νία αποχώρησης'].fillna(today) - df['Ημ/νία πρόσληψης']).dt.days // 365
if 'Ημ/νία αποχώρησης' in df.columns:
    df['Attrition'] = df['Ημ/νία αποχώρησης'].notnull().astype(int)

df.rename(columns={
    'Αριθμός μητρώου': 'Registry Number', 'Φύλο': 'Gender', 'Ηλικία': 'Age',
    'Περιγραφή Αιτ. Αποχώρησης': 'Departure Reason Description', 'Ονομαστικός μισθός': 'Nominal Salary',
    'Σχέση Εργασίας': 'Work Relationship', 'Περιγραφή Υποκαταστήματος': 'City',
    'Διεύθυνση': 'Division', 'Ιδιότητα Προσωπικού': 'Job Property',
    'Θέση εργασίας': 'Job Position', 'GRADE': 'Grade', 'ΤΜΗΜΑ': 'Department'
}, inplace=True)

# Filters: Only Indefinite contracts and Voluntary departures
df = df[df['Work Relationship'] == 'ΑΟΡΙΣΤΟΥ ΧΡΟΝΟΥ']
df = df[(df['Departure Reason Description'] == 'VOLUNTARY DEPARTURE') | (df['Departure Reason Description'].isnull())]
df.loc[df['Department'].astype(str).str.contains('ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ', na=False), 'Attrition'] = 0
df['Gender'] = df['Gender'].replace({1: 'Male', 2: 'Female'})
df = df[(df['Departure Date'] > '2018-12-31') | (df['Departure Date'].isnull())]

# Salary & Grade Cleaning
df['Nominal Salary'] = df['Nominal Salary'].str.replace(',', '.', regex=False)
df['Nominal Salary'] = pd.to_numeric(df['Nominal Salary'], errors='coerce')
df['Nominal Salary'].fillna(df['Nominal Salary'].median(), inplace=True)
df['Job Property'] = df['Job Property'].fillna('OPERATIONAL')
df['Grade'] = df['Grade'].replace({'99999': '0.9', '0,1': '0.99'}).astype(float)

# Save active employees for final prediction before dropping columns
active_mask = df['Attrition'] == 0
df_active_raw = df[active_mask].copy()

# Prepare Modeling Data
df_ml = df.drop(columns=['Departure Date', 'Ημ/νία γέννησης', 'Ημ/νία πρόσληψης', 'Work Relationship', 'Departure Reason Description'])
categorical_columns = ['Gender', 'City', 'Division', 'Job Property', 'Job Position', 'Department']
df_transformed = pd.get_dummies(df_ml, columns=categorical_columns, drop_first=True)

X = df_transformed.drop(columns=['Attrition', 'Registry Number'])
y = df_transformed['Attrition']

# --- 2. Train/Test Split (The Fix for Leakage) ---
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Calculate Class Weights
counts = np.bincount(y_train)
scale_pos_weight_value = counts[0] / counts[1]

# --- 3. Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight_value, random_state=42, use_label_encoder=False, eval_metric='logloss')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=kf, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

# --- 4. Evaluation on Unseen Test Data ---
y_pred = best_xgb.predict(X_test)
print("\n--- Test Set Performance ---")
print(classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_xgb, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.show()

# --- 5. Prediction for Currently Active Employees ---
# Aligning active employees with training columns
X_active = df_transformed[active_mask].drop(columns=['Attrition', 'Registry Number'])
X_active_scaled = scaler.transform(X_active)

probs = best_xgb.predict_proba(X_active_scaled)[:, 1]
preds = (probs > 0.5).astype(int)

df_active_raw['Attrition_Probability'] = probs
df_active_raw['Predicted_Attrition'] = preds

# --- 6. SHAP Interpretability ---
# Use TreeExplainer for XGBoost (much faster)
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train)

print("\nGenerating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_train_raw)


# --- 7. Final Results Output ---
likely_to_attrite = df_active_raw[df_active_raw['Predicted_Attrition'] == 1]
print(f"\nTotal Active Employees: {len(df_active_raw)}")
print(f"Predicted to leave next year: {len(likely_to_attrite)}")
print("\nTop 5 Risks:")
print(likely_to_attrite[['Registry Number', 'Division', 'Attrition_Probability']].sort_values(by='Attrition_Probability', ascending=False).head())
