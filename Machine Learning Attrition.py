import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import shap
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# Load the dataset
file_path = r"C:\Users\sy.papadopoulos\OneDrive - Alumil S.A\Desktop\ml attrition data latest.csv"

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-7', sep=';')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp1253', sep=';')
except pd.errors.ParserError as e:
    print("ParserError:", e)
    exit()

# Strip column names of extra whitespace
df.columns = df.columns.str.strip()

# Convert date columns to datetime
date_columns = ['Ημ/νία γέννησης', 'Ημ/νία πρόσληψης', 'Ημ/νία αποχώρησης']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce', dayfirst=True)

# Create new features (Age, Tenure, Attrition)
today = datetime.today()
if 'Ημ/νία πρόσληψης' in df.columns:
    df['Tenure'] = (df['Ημ/νία αποχώρησης'].fillna(today) - df['Ημ/νία πρόσληψης']).dt.days // 365

if 'Ημ/νία αποχώρησης' in df.columns:
    df['Attrition'] = df['Ημ/νία αποχώρησης'].notnull().astype(int)


# Rename columns
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

# Keep only indefinite-term contracts and voluntary departures
df = df[df['Work Relationship'] == 'ΑΟΡΙΣΤΟΥ ΧΡΟΝΟΥ']
df = df[(df['Departure Reason Description'] == 'VOLUNTARY DEPARTURE') | (df['Departure Reason Description'].isnull())]
# Update 'Attrition' column where 'Department' contains the word "ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ"
df.loc[df['Department'].astype(str).str.contains('ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ', na=False), 'Attrition'] = 0

# Convert Gender column values
df['Gender'] = df['Gender'].replace({1: 'Male', 2: 'Female'})

# Filter rows where 'Departure Date' is either greater than 2018-12-31 or is null
df = df[(df['Departure Date'] > '2018-12-31') | (df['Departure Date'].isnull())]

# Clean up Nominal Salary
df['Nominal Salary'] = df['Nominal Salary'].str.replace(',', '.', regex=False)
df['Nominal Salary'] = pd.to_numeric(df['Nominal Salary'], errors='coerce')
df['Nominal Salary'].fillna(df['Nominal Salary'].median(), inplace=True)

# Replace NaN values
df['Job Property'] = df['Job Property'].fillna('OPERATIONAL')

# Replace specific values in the 'Grade' column
df['Grade'] = df['Grade'].replace({
    '99999': '0.9',  # Replace '99999' with '0.9'
    '0,1': '0.99'    # Replace '0,1' with '0.99'
})
df['Grade'] = df['Grade'].astype(float)

# Store 'Registry Number' separately for mapping predictions later
registry_numbers = df['Registry Number']

# Drop unnecessary columns
df.drop(columns=['Departure Date', 'Hire Date', 'Work Relationship', 'Registry Number', 'Departure Reason Description'], inplace=True)

# Convert categorical columns to dummies
categorical_columns = ['Gender', 'City', 'Division', 'Job Property', 'Job Position', 'Department']
df_transformed = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Prepare data for modeling
X = df_transformed.drop(columns=['Attrition'])
y = df_transformed['Attrition']
object_cols = X.select_dtypes(include=['object']).columns
print("Object columns in X:", object_cols)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Fine-tune XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.01, 0.1],
    'reg_lambda': [0.01, 0.1]
}



xgb = XGBClassifier(scale_pos_weight=class_weights[1] / class_weights[0], random_state=42)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_scaled, y)

# Best model and predictions
best_xgb = grid_search.best_estimator_
y_proba = best_xgb.predict_proba(X_scaled)[:, 1]
y_pred = best_xgb.predict(X_scaled)

# Combine predictions with 'Registry Number'
results = pd.DataFrame({
    'Registry Number': registry_numbers,
    'Attrition_Probability': y_proba,
    'Predicted_Attrition': y_pred
})



# Calculate total predicted attrition
total_predicted_attrition = results['Predicted_Attrition'].sum()
print(f"Total Predicted Attrition: {total_predicted_attrition}")

# View employees likely to leave
likely_to_leave = results[results['Predicted_Attrition'] == 1]
print("Employees Likely to Leave:")
print(likely_to_leave)

# Step 1: Filter active employees (Departure Date is null)
active_employees = df[df['Attrition'] == 0]  # Attrition == 0 means still active
active_employees['Registry Number'] = registry_numbers[df['Attrition'] == 0]  # Add Registry Number back

# Step 2: Ensure consistent features with the trained model
active_employees_transformed = pd.get_dummies(active_employees, columns=categorical_columns, drop_first=True)

# Add missing columns (from training data) as 0 to maintain consistency
missing_cols = set(X.columns) - set(active_employees_transformed.columns)
for col in missing_cols:
    active_employees_transformed[col] = 0

# Reorder columns to match the training data
active_employees_transformed = active_employees_transformed[X.columns]

# Step 3: Scale the features using the same scaler
active_employees_scaled = scaler.transform(active_employees_transformed)

# Step 4: Predict attrition for next year
active_employees['Attrition_Probability'] = best_xgb.predict_proba(active_employees_scaled)[:, 1]
active_employees['Predicted_Attrition'] = best_xgb.predict(active_employees_scaled)

# Step 5: Filter employees likely to attrite
likely_to_attrite_next_year = active_employees[active_employees['Predicted_Attrition'] == 1]

# Display the number of employees predicted to attrite
num_likely_to_attrite = len(likely_to_attrite_next_year)
print(f"Number of employees predicted to attrite next year: {num_likely_to_attrite}")

# Display employees likely to attrite
print("Employees Likely to Attrite Next Year:")
print(likely_to_attrite_next_year[['Registry Number', 'Attrition_Probability', 'Predicted_Attrition']])

# Optional: Group by Division or Department to analyze
attrition_by_division = likely_to_attrite_next_year.groupby('Division')['Predicted_Attrition'].sum().sort_values(ascending=False)

print("Predicted Attrition by Division:")
print(attrition_by_division)

# Print the Registry Numbers of employees likely to attrite by Division
print("\nRegistry Numbers of employees likely to attrite by Division:")
attrition_by_division_list = likely_to_attrite_next_year.groupby('Division')['Registry Number'].apply(list)

for division_name, reg_list in attrition_by_division_list.items():
    print(f"\nDivision: {division_name}")
    for reg in reg_list:
        print(f"   Registry Number: {reg}")

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

historical_attrition_rate = df['Attrition'].mean()
predicted_attrition_rate = likely_to_attrite_next_year.shape[0] / active_employees.shape[0]

print(f"Historical Attrition Rate: {historical_attrition_rate:.2%}")
print(f"Predicted Attrition Rate: {predicted_attrition_rate:.2%}")



# Verify and fix column consistency
for col in set(df_transformed.columns) - set(X.columns):
    X[col] = 0
X = X[df_transformed.columns]

# Ensure all columns are numeric and fill missing values
X.fillna(0, inplace=True)
X = X.astype(float)

# Initialize SHAP
explainer = shap.Explainer(best_xgb, X)
shap_values = explainer(X)

# Plot SHAP summary
shap.summary_plot(shap_values, X)


