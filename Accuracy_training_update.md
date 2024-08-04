# Football Match Prediction

This project aims to predict the outcome and scores of football matches using machine learning models. The application utilizes an ensemble of multiple classifiers and is capable of predicting both match outcomes and exact scores for home and away teams.

## Features

- Predicts match outcomes (home win, away win, draw)
- Predicts exact scores for home and away teams
- Displays probabilities of each possible outcome
- Provides head-to-head history between selected teams
- Shows recent match history for both home and away teams

## Model Performance

### Accuracy

- **Overall Accuracy**: 73.03%

### F1-Scores

- **Home Wins**: 79%
- **Away Wins**: 75%
- **Draws**: 59%

### Classification Report

| Outcome          | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Away Win         | 74%       | 76%    | 75%      | 1213    |
| Draw             | 65%       | 54%    | 59%      | 1028    |
| Home Win         | 76%       | 82%    | 79%      | 1727    |
| **Macro Avg**    | 72%       | 71%    | 71%      | 3968    |
| **Weighted Avg** | 73%       | 73%    | 73%      | 3968    |

### Mean Absolute Error (MAE) for Score Predictions

- **Home Goals Prediction MAE**: 0.6316
- **Away Goals Prediction MAE**: 0.5565

### Confusion Matrix

- **Away Win**:
  - True: 1213
  - Predicted correctly: 925
- **Draw**:
  - True: 1028
  - Predicted correctly: 549
- **Home Win**:
  - True: 1727
  - Predicted correctly: 1425

## Model Components

### Ensemble Model

The ensemble model combines multiple machine learning algorithms:

- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM) Classifier

### Voting Mechanism

- Uses "soft" voting: averages the predicted probabilities from each model and selects the class with the highest average probability as the final prediction.

### Feature Processing

- **Imputation**: Filling in missing values using the mean of the column.
- **Scaling**: Standardizing the features so they are on the same scale.
- **Feature Selection**: Using a Random Forest to select the most important features.

### Key Features

- Team strengths (offensive and defensive)
- Expected goals (xG) for home and away teams
- Head-to-head statistics
- Recent form of both teams

## Flask API

The Flask API provides endpoints for making predictions and fetching team lists:

### Endpoints

- `/`: Renders the homepage.
- `/predict`: Accepts POST requests with `home_team` and `away_team` to predict match outcomes and scores.
- `/teams`: Returns a list of unique teams.

### Usage

1. Run the Flask app:

   ```sh
   python app.py
   ```
