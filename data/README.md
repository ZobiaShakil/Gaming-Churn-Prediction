## gaming_churn_synthetic.csv
This file contains synthetic data for a churn prediction project. The data includes columns such as:
- **player_id**: A unique identifier for each player.
- **total_playtime_hours**: The total number of hours a player has spent in the game.
- **last_login_days_ago**: The number of days since the player last logged in.
- **in_app_purchases**: The amount spent on in-app purchases.
- **churn_status**: Indicates whether the player has stopped playing (1 for churned, 0 for active).

## gaming_churn_preprocessed.csv
Contains the initial cleaned and preprocessed version of the gaming user data. This includes handling of missing values, basic transformations, and feature selection.

## gaming_churn_final.csv
This is the final dataset used for model training and evaluation. It includes engineered features such as:

-tenure_bucket

-engagement_score
These features help improve model performance and interpretability.

🧩 Target Variable:
churn_status:
Binary column indicating if a user has churned (1) or is still active (0).
