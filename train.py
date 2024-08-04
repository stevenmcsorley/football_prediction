import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# Load the data
df = pd.read_csv('football_match_data.csv')

# Print info about the dataset
print("Dataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst few rows:")
print(df.head())

# Feature engineering
df['goal_difference'] = df['home_goals'] - df['away_goals']
df['result'] = df['goal_difference'].apply(lambda x: 'home_win' if x > 0 else ('away_win' if x < 0 else 'draw'))
df['xG_difference'] = df['home_xG'] - df['away_xG']

# Create team strength features
team_strength = df.groupby('home_team')['home_goals'].mean().to_dict()
df['home_team_strength'] = df['home_team'].map(team_strength)
df['away_team_strength'] = df['away_team'].map(team_strength)

# Prepare features and target
features = ['home_team_strength', 'away_team_strength', 'home_xG', 'away_xG']
X = df[features]
y = df['result']

# Print info about features
print("\nFeature info:")
print(X.info())

print("\nFeature description:")
print(X.describe())

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a pipeline with imputer, scaler, and model
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Define models
models = {
    'Logistic Regression': create_pipeline(LogisticRegression(random_state=42, multi_class='ovr')),
    'SVM': create_pipeline(SVC(random_state=42)),
    'Random Forest': create_pipeline(RandomForestClassifier(random_state=42)),
    'XGBoost': create_pipeline(XGBClassifier(random_state=42))
}

# Train and evaluate models
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the best model (in this case, we'll save XGBoost)
best_model = models['XGBoost']
joblib.dump(best_model, 'football_prediction_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\nBest model (XGBoost) and label encoder saved.")