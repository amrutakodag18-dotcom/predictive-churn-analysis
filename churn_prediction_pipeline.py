import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load Dataset
file_name = 'customer_churn_model_v2.csv'
df = pd.read_csv(file_name)

print("--- 1. Dataset Overview ---")
print(f"Total Customer Records: {len(df)}")
print(df.head(3))

# Map target variable (Churn: Yes -> 1, No -> 0)
df['Churn_Target'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Features and Target split
X = df.drop(columns=['CustomerID', 'Churn', 'Churn_Target'])
y = df['Churn_Target']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric Features ({len(num_cols)}): {num_cols}")
print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")

# 2. Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# Preprocessing Pipeline (One-Hot Encoding for categorical, MinMaxScaler for numerical)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

# 3. Train Classifier (Logistic Regression & Random Forest)
log_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\n--- Training Logistic Regression Model ---")
log_pipeline.fit(X_train, y_train)

print("--- Training Random Forest Classifier ---")
rf_pipeline.fit(X_train, y_train)

# 4. Model Evaluation
y_pred_log = log_pipeline.predict(X_test)
y_prob_log = log_pipeline.predict_proba(X_test)[:, 1]

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n==========================================")
print("       LOGISTIC REGRESSION REPORT         ")
print("==========================================")
print(classification_report(y_test, y_pred_log))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_log):.4f}")

print("\n==========================================")
print("        RANDOM FOREST REPORT              ")
print("==========================================")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

# Plot ROC-AUC Curves
plt.figure(figsize=(8, 6))
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_log):.3f})", color="blue", lw=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", color="green", lw=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve Comparison for Churn Prediction")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("roc_auc_curve.png")
print("\nROC-AUC curve plot saved as 'roc_auc_curve.png'.")

# 5. Export Customer Churn Risk Scores
# Using Random Forest probabilities on the test set
results_df = X_test.copy()
results_df['CustomerID'] = df.loc[X_test.index, 'CustomerID']
results_df['Actual_Churn'] = y_test
results_df['Predicted_Churn_Prob'] = y_prob_rf
results_df['Risk_Level'] = pd.cut(
    results_df['Predicted_Churn_Prob'],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Export predictions
output_filename = 'customer_churn_risk_scores.csv'
results_df[['CustomerID', 'Actual_Churn', 'Predicted_Churn_Prob', 'Risk_Level']].to_csv(output_filename, index=False)
print(f"Success! Customer churn risk predictions exported to '{output_filename}'.")