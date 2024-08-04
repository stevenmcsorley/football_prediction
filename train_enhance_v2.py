import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('football_match_data.csv')

# Feature engineering
def engineer_features(df):
    df['goal_difference'] = df['home_goals'] - df['away_goals']
    df['xG_difference'] = df['home_xG'] - df['away_xG']
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort the dataframe by date
    df = df.sort_values('date')
    
    # Calculate team strengths based on rolling average
    window = 10
    df['home_team_strength'] = df.groupby('home_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['away_team_strength'] = df.groupby('away_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['home_team_defense'] = df.groupby('home_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['away_team_defense'] = df.groupby('away_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Calculate recent form (last 5 matches)
    df['home_form'] = df.groupby('home_team')['goal_difference'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['away_form'] = df.groupby('away_team')['goal_difference'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Head-to-head performance
    def get_h2h_stats(group):
        home_wins = (group['home_goals'] > group['away_goals']).sum()
        away_wins = (group['home_goals'] < group['away_goals']).sum()
        draws = (group['home_goals'] == group['away_goals']).sum()
        total_matches = len(group)
        return pd.Series({
            'h2h_home_win_rate': home_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_away_win_rate': away_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_draw_rate': draws / total_matches if total_matches > 0 else 0.5
        })

    h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats).reset_index()
    df = pd.merge(df, h2h_stats, on=['home_team', 'away_team'], how='left')
    
    # Create 'result' column
    df['result'] = np.select(
        [df['goal_difference'] > 0, df['goal_difference'] < 0, df['goal_difference'] == 0],
        ['home_win', 'away_win', 'draw']
    )
    
    # Additional features
    df['goal_ratio'] = df['home_goals'] / (df['away_goals'] + 1e-5)  # Avoid division by zero
    df['xG_ratio'] = df['home_xG'] / (df['away_xG'] + 1e-5)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    return df

df = engineer_features(df)

# Handle NaN values in the target variables
df = df.dropna(subset=['home_goals', 'away_goals'])

# Prepare features and target
features = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
            'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
            'goal_ratio', 'xG_ratio', 'day_of_week', 'month']
X = df[features]
y_result = df['result']
y_home_goals = df['home_goals']
y_away_goals = df['away_goals']

# Encode the target variable for result prediction
le = LabelEncoder()
y_result_encoded = le.fit_transform(y_result)

# Time-based split
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_result_train, y_result_test = y_result_encoded[:train_size], y_result_encoded[train_size:]
y_home_goals_train, y_home_goals_test = y_home_goals[:train_size], y_home_goals[train_size:]
y_away_goals_train, y_away_goals_test = y_away_goals[:train_size], y_away_goals[train_size:]

# Create a pipeline with imputer, scaler, feature selection, and model
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))),
        ('model', model)
    ])

# Define models with hyperparameter distributions for random search
rf_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None] + list(randint(10, 50).rvs(4)),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'feature_selection__max_features': randint(2, len(features))
}

xgb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__colsample_bytree': uniform(0.5, 0.5),
    'feature_selection__max_features': randint(2, len(features))
}

gb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'feature_selection__max_features': randint(2, len(features))
}

models = {
    'RandomForest': (create_pipeline(RandomForestClassifier(random_state=42)), rf_params),
    'XGBoost': (create_pipeline(XGBClassifier(random_state=42)), xgb_params),
    'GradientBoosting': (create_pipeline(GradientBoostingClassifier(random_state=42)), gb_params),
}

# Function to print feature importance
def print_feature_importance(pipeline, feature_names):
    # Get the final estimator (model) from the pipeline
    model = pipeline.named_steps['model']
    
    if hasattr(model, 'feature_importances_'):
        # Get the feature selector from the pipeline
        feature_selector = pipeline.named_steps['feature_selection']
        # Get the mask of selected features
        feature_mask = feature_selector.get_support()
        # Filter the feature names
        selected_features = [f for f, selected in zip(feature_names, feature_mask) if selected]
        
        importances = model.feature_importances_
        
        # Sort features by importance
        feature_importance = sorted(zip(importances, selected_features), reverse=True)
        
        print("Top 10 most important features:")
        for i, (importance, feature) in enumerate(feature_importance[:10], 1):
            print(f"{i}. {feature} ({importance:.6f})")
    else:
        print("This model doesn't have feature importances.")

# Load the previously tuned SVC model from the existing ensemble
print("Loading previously tuned SVC model...")
ensemble_model = joblib.load('football_prediction_ensemble.joblib')
svc_model = [model for name, model in ensemble_model.named_estimators_.items() if name == 'SVC'][0]

# Perform random search for each model (except SVC)
best_models = {}
for name, (pipeline, params) in models.items():
    print(f"Tuning {name}...")
    random_search = RandomizedSearchCV(pipeline, params, n_iter=50, cv=5, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X_train, y_result_train)
    best_models[name] = random_search.best_estimator_
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Best score for {name}: {random_search.best_score_}")
    print(f"\nFeature importance for {name}:")
    print_feature_importance(random_search.best_estimator_, features)

# Add the pre-tuned SVC model to the best_models dictionary
best_models['SVC'] = svc_model

# Create ensemble model for result prediction
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

# Train and evaluate the ensemble model for result prediction
ensemble.fit(X_train, y_result_train)
y_result_pred = ensemble.predict(X_test)
result_accuracy = accuracy_score(y_result_test, y_result_pred)
print("\nEnsemble Model Results:")
print(f"Accuracy: {result_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_result_test, y_result_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_result_test, y_result_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Cross-validation
cv_scores = cross_val_score(ensemble, X, y_result_encoded, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train models for score prediction
score_models = {
    'home_goals': RandomForestClassifier(n_estimators=100, random_state=42),
    'away_goals': RandomForestClassifier(n_estimators=100, random_state=42)
}

score_models['home_goals'].fit(X_train, y_home_goals_train)
score_models['away_goals'].fit(X_train, y_away_goals_train)

# Evaluate score prediction models
y_home_goals_pred = score_models['home_goals'].predict(X_test)
y_away_goals_pred = score_models['away_goals'].predict(X_test)
home_goals_mae = np.mean(np.abs(y_home_goals_test - y_home_goals_pred))
away_goals_mae = np.mean(np.abs(y_away_goals_test - y_away_goals_pred))
print(f"\nHome Goals Prediction MAE: {home_goals_mae:.4f}")
print(f"Away Goals Prediction MAE: {away_goals_mae:.4f}")

# Save the ensemble model, label encoder, and score models
joblib.dump(ensemble, 'football_prediction_ensemble.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(score_models['home_goals'], 'home_goals_model.joblib')
joblib.dump(score_models['away_goals'], 'away_goals_model.joblib')

print("\nEnsemble model, label encoder, and score prediction models saved.")