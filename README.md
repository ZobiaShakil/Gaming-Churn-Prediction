# Gaming-Churn-Prediction
 Predict gaming customer churn using synthetic data. This project demonstrates an end-to-end machine learning pipeline from EDA, data cleaning, and feature engineering to training multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with hyperparameter tuning and interpretability using SHAP.

### Data Description
The dataset contains synthetic data with the following key features:

player_id: Unique identifier for each player.

total_playtime_hours: Total hours played.

last_login_days_ago: Days since the player’s last login.

in_app_purchases: Number or amount of in-app purchases.

churn_status: 0 indicates active, 1 indicates churned.

Engineered Features:

tenure_bucket: Categorizes players based on playtime (e.g., Low, Medium, High).

engagement_score: Calculated as in_app_purchases divided by total_playtime_hours.

### Methodology
Data Preprocessing:

Load the dataset, handle missing values (e.g., fill missing tenure_bucket with "Unknown"), and scale numerical features.

Feature Engineering:

Create additional features (e.g., tenure buckets and engagement scores) to enrich the dataset.

 Model Training:

Train multiple models including Logistic Regression, Decision Tree, Random Forest, and XGBoost using cross-validation.

Evaluate models based on accuracy and F1-score.

Hyperparameter Tuning:

Use GridSearchCV to fine-tune parameters for the best-performing model.

Model Interpretation:

Use SHAP to visualize feature importance and understand decision thresholds.

### Results
After evaluating models with cross-validation and performing hyperparameter tuning, the best model was selected based on F1-score. The final tuned model achieved around 61% accuracy and an F1 score of approximately 0.235 on the test set. SHAP analysis provided insights into which features (such as playtime and engagement score) most strongly influenced the predictions.
