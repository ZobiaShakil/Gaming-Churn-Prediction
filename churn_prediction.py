# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling and Evaluation libraries 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# For SHAP explainability
import shap

# =========================
# 2. Load the Preprocessed & Engineered Data
# =========================
# Use one of the files; if "gaming_churn_final.csv" is the final dataset, load it:
df = pd.read_csv(r"C:\Users\zobia\OneDrive\AI-ML\Projects\CustomerChurnPred\gaming_churn_final.csv")

# Fix missing values in tenure_bucket
df['tenure_bucket'] = df['tenure_bucket'].fillna("Unknown")

# Display basic information
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())

# =========================
# 3. Define Features and Target
# =========================
# Using the numerical and engineered features (excluding the categorical 'tenure_bucket' for now)
features = ['total_playtime_hours', 'last_login_days_ago', 'in_app_purchases', 'engagement_score']
X = df[features]
y = df['churn_status']  # 0 for active, 1 for churned

# =========================
# 4. Split Data into Training and Testing Sets
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 5. Model Training and Cross-Validation
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

scoring_metrics = {"Accuracy": "accuracy", "F1": make_scorer(f1_score)}

cv_results = {}
print("Cross-Validation Results:")
for name, model in models.items():
    scores = {}
    for metric, scorer in scoring_metrics.items():
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
        scores[metric] = cv_score.mean()
    cv_results[name] = scores
    print(f"{name} - Accuracy: {scores['Accuracy']:.3f}, F1: {scores['F1']:.3f}")

# =========================
# 6. Select the Best Model
# =========================
best_model_name = max(cv_results, key=lambda m: cv_results[m]['F1'])
print("\nBest Model Based on CV F1-Score:", best_model_name)
best_model = models[best_model_name]

# =========================
# 7. Hyperparameter Tuning for the Best Model (using GridSearchCV)
# =========================
if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
elif best_model_name == "XGBoost":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
elif best_model_name == "Decision Tree":
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
elif best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
else:
    param_grid = {}

print("\nHyperparameter Tuning for", best_model_name)
grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1-Score:", grid_search.best_score_)

tuned_model = grid_search.best_estimator_

# =========================
# 8. Evaluate Tuned Model on Test Set
# =========================
y_pred = tuned_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
print("\nTest Set Evaluation:")
print("Accuracy:", test_accuracy)
print("F1 Score:", test_f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 9. Model Interpretation with SHAP
# =========================
explainer = shap.TreeExplainer(tuned_model)
shap_values = explainer.shap_values(X_train)

# Plot SHAP summary plot for the first class (churned customers)
# If you are doing binary classification:
shap_values_for_class_1 = shap_values[:, :, 1]

# Generate a SHAP summary bar plot
# use the shap_values for the correct class instead of the incorrect dimension
shap.summary_plot(shap_values_for_class_1, X_train, plot_type="bar")
# =========================
# 10. Save the Final Tuned Model and Preprocessed Data
# =========================
"""
import joblib
joblib.dump(tuned_model, "final_tuned_churn_model.pkl")
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
print("\nFinal tuned model and preprocessed data saved.")
"""
