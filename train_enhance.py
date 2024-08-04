import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    
    # Calculate team strengths based on average goals scored and conceded
    home_strength = df.groupby('home_team')['home_goals'].mean()
    away_strength = df.groupby('away_team')['away_goals'].mean()
    home_defense = df.groupby('home_team')['away_goals'].mean()
    away_defense = df.groupby('away_team')['home_goals'].mean()
    
    df['home_team_strength'] = df['home_team'].map(home_strength)
    df['away_team_strength'] = df['away_team'].map(away_strength)
    df['home_team_defense'] = df['home_team'].map(home_defense)
    df['away_team_defense'] = df['away_team'].map(away_defense)
    
    # Calculate recent form (last 5 matches)
    df['home_form'] = df.groupby('home_team')['goal_difference'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['away_form'] = df.groupby('away_team')['goal_difference'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    
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
    
    return df

df = engineer_features(df)

# Handle NaN values in the target variables
df = df.dropna(subset=['home_goals', 'away_goals'])

# Prepare features and target
features = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
            'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate']
X = df[features]
y_result = df['result']
y_home_goals = df['home_goals']
y_away_goals = df['away_goals']

# Encode the target variable for result prediction
le = LabelEncoder()
y_result_encoded = le.fit_transform(y_result)

# Split the data
X_train, X_test, y_result_train, y_result_test = train_test_split(X, y_result_encoded, test_size=0.2, random_state=42)
_, _, y_home_goals_train, y_home_goals_test = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)
_, _, y_away_goals_train, y_away_goals_test = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)

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
    'model__min_samples_leaf': randint(1, 10)
}

xgb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.6, 0.4)
}

gb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.6, 0.4),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10)
}

svc_params = {
    'model__C': uniform(0.1, 10),
    'model__kernel': ['rbf', 'poly'],
    'model__gamma': uniform(0.01, 1)
}

models = {
    'RandomForest': (create_pipeline(RandomForestClassifier(random_state=42)), rf_params),
    'XGBoost': (create_pipeline(XGBClassifier(random_state=42)), xgb_params),
    'GradientBoosting': (create_pipeline(GradientBoostingClassifier(random_state=42)), gb_params),
    'SVC': (create_pipeline(SVC(random_state=42, probability=True)), svc_params)
}

# Perform random search for each model
best_models = {}
for name, (pipeline, params) in models.items():
    print(f"Tuning {name}...")
    random_search = RandomizedSearchCV(pipeline, params, n_iter=50, cv=5, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X_train, y_result_train)
    best_models[name] = random_search.best_estimator_
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Best score for {name}: {random_search.best_score_}")

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
