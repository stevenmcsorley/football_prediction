# Football Match Prediction System

This project is a web-based application that predicts the outcomes of football matches using machine learning models and Expected Goals (xG) data.

## Overview

The system consists of three main components:

1. Data scraping script
2. Machine learning model training script
3. Flask web application for predictions

## Data Overview

- Dataset size: 19,938 entries
- Features: home_team_strength, away_team_strength, home_xG, away_xG
- Target: Match outcome (home_win, away_win, draw)
- Note: 101 missing values in each of 'home_goals', 'away_goals', 'home_xG', and 'away_xG' columns

## Model Performance

We trained and evaluated four different models:

1. Logistic Regression: 60.38% accuracy
2. SVM: 60.33% accuracy
3. Random Forest: 64.17% accuracy
4. XGBoost: 62.26% accuracy

The Random Forest model performed the best, followed by XGBoost.

### Detailed Analysis (Random Forest)

- Away Win: Precision 0.64, Recall 0.70, F1-score 0.67
- Draw: Precision 0.51, Recall 0.35, F1-score 0.41
- Home Win: Precision 0.69, Recall 0.78, F1-score 0.73

Observations:

- The model is best at predicting home wins, followed by away wins.
- It struggles the most with predicting draws, which is common in football prediction models.
- The overall accuracy of 64.17% is decent for a football prediction model, considering the inherent unpredictability of the sport.

## Setup and Usage

1. Clone the repository:

   ```
   git clone [repository-url]
   cd football-prediction-system
   ```

2. Install required packages:

   ```
   pip install -r requirements.txt
   ```

3. Run the data scraping script:

   ```
   python scrape.py
   ```

4. Train the model:

   ```
   python train.py
   ```

5. Start the Flask application:

   ```
   python app.py
   ```

6. Open a web browser and navigate to `http://localhost:5000` to use the prediction interface.

## Files in the Project

- `scrape.py`: Script for scraping football match data
- `train.py`: Script for training and evaluating machine learning models
- `app.py`: Flask application for serving predictions
- `templates/index.html`: HTML template for the web interface
- `football_prediction_model.joblib`: Saved machine learning model
- `label_encoder.joblib`: Saved label encoder for outcome categories

## How to Use the Prediction Interface

1. Select the home team and away team from the dropdowns.
2. Enter the expected xG (Expected Goals) for both the home and away teams.
   - Typical range: 0.5 (low) to 2.5+ (exceptional)
3. Click "Predict" to get the match outcome prediction and probabilities.

## Future Improvements

- Implement the Random Forest model instead of XGBoost for potentially better performance.
- Explore feature engineering to improve prediction accuracy, especially for draws.
- Experiment with ensemble methods, combining predictions from multiple models.
- Perform hyperparameter tuning for the Random Forest and XGBoost models.

## Note on xG Data

Expected Goals (xG) is a statistical measure of the quality of chances created in a football match. If you're unsure about xG values, refer to the "Learn more about xG" section in the web interface for detailed explanations and examples.

## Disclaimer

This tool is for informational purposes only. Please use responsibly and do not use it for gambling or any illegal activities.
