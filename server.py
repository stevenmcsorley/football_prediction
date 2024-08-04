from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = joblib.load('football_prediction_model.joblib')
le = joblib.load('label_encoder.joblib')

# Load team strength data (you might want to update this periodically)
df = pd.read_csv('football_match_data.csv')
team_strength = df.groupby('home_team')['home_goals'].mean().to_dict()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    home_xG = float(data['home_xG'])
    away_xG = float(data['away_xG'])
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'home_team_strength': [team_strength.get(home_team, np.mean(list(team_strength.values())))],
        'away_team_strength': [team_strength.get(away_team, np.mean(list(team_strength.values())))],
        'home_xG': [home_xG],
        'away_xG': [away_xG]
    })
    
    # Make prediction
    prediction_encoded = model.predict(input_data)
    prediction = le.inverse_transform(prediction_encoded)[0]
    
    probabilities = model.predict_proba(input_data)[0]
    probability_dict = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probabilities)}
    
    return jsonify({
        'prediction': prediction,
        'probabilities': probability_dict
    })

@app.route('/team_strength', methods=['GET'])
def get_team_strength():
    return jsonify(team_strength)

@app.route('/teams', methods=['GET'])
def get_teams():
    teams = sorted(team_strength.keys())
    return jsonify(teams)

if __name__ == '__main__':
    app.run(debug=True)