import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# ----------------------------
# Load and Prep
# ----------------------------
df = pd.read_csv("train_filtered.csv")
X = df[["lines_of_code", "cyclomatic_complexity", "num_functions", "num_classes"]]
y = df["defect"]

# Calculate the scale factor (Total Negatives / Total Positives)
# Since '0' is your minority, we want to weight it higher.
counter = y.value_counts()
weight_ratio = counter[1] / counter[0] # ~32.7

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Optimized XGBoost
# ----------------------------
model = XGBClassifier(
    n_estimators=200,          # More trees for complex boundaries
    max_depth=6,               # Slightly deeper to catch feature interactions
    learning_rate=0.05,        # Slower learning rate for better convergence
    scale_pos_weight=1,        # We'll use a different approach for minority '0'
    eval_metric="aucpr",       # Focus on Precision-Recall AUC instead of LogLoss
    random_state=42
)

# Alternative: If you want to focus on the rare '0' class, 
# it is often better to use sample_weights during fit:
sample_weights = np.where(y_train == 0, weight_ratio, 1)

model.fit(X_train, y_train, sample_weight=sample_weights)

# ----------------------------
# Evaluation logic
# ----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
# Instead of a hardcoded 0.987, let's use the median of probabilities 
# as a starting point to see a balanced confusion matrix
optimal_threshold =np.median(y_prob)
y_pred = (y_prob >= optimal_threshold).astype(int)

print(f"--- Evaluation (Threshold: {optimal_threshold:.4f}) ---")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))