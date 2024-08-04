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

# Prepare features and target
features = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
            'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate']
X = df[features]
y = df['result']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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
    random_search.fit(X_train, y_train)
    best_models[name] = random_search.best_estimator_
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Best score for {name}: {random_search.best_score_}")

# Create ensemble model
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

# Train and evaluate the ensemble model
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nEnsemble Model Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Save the ensemble model and label encoder
joblib.dump(ensemble, 'football_prediction_ensemble.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\nEnsemble model and label encoder saved.")

# Optional: If you want to save individual models as well
for name, model in best_models.items():
    joblib.dump(model, f'football_prediction_{name.lower()}.joblib')
    print(f"{name} model saved.")

# Feature importance analysis
rf_model = best_models['RandomForest']
feature_selector = rf_model.named_steps['feature_selection']
selected_feature_indices = feature_selector.get_support(indices=True)
selected_features = [features[i] for i in selected_feature_indices]

feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(x='feature', y='importance', kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nFeature importance plot saved as 'feature_importance.png'")

# Learning curves
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve for the ensemble model
plot_learning_curve(ensemble, "Learning Curve for Ensemble Model", X, y_encoded, ylim=(0.1, 1.01), cv=5, n_jobs=-1)
plt.savefig('learning_curve.png')
plt.close()

print("\nLearning curve plot saved as 'learning_curve.png'")