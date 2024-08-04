from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime

app = Flask(__name__)

# Load the trained ensemble model and label encoder
ensemble_model = joblib.load('football_prediction_ensemble.joblib')
le = joblib.load('label_encoder.joblib')

# Load team strength data and other necessary data
df = pd.read_csv('football_match_data.csv')

# Calculate team strengths and defenses
@lru_cache(maxsize=None)
def get_team_stats():
    home_strength = df.groupby('home_team')['home_goals'].mean().to_dict()
    away_strength = df.groupby('away_team')['away_goals'].mean().to_dict()
    home_defense = df.groupby('home_team')['away_goals'].mean().to_dict()
    away_defense = df.groupby('away_team')['home_goals'].mean().to_dict()
    return home_strength, away_strength, home_defense, away_defense

# Calculate head-to-head statistics
@lru_cache(maxsize=None)
def get_h2h_stats():
    def calculate_h2h(group):
        home_wins = (group['home_goals'] > group['away_goals']).sum()
        away_wins = (group['home_goals'] < group['away_goals']).sum()
        draws = (group['home_goals'] == group['away_goals']).sum()
        total_matches = len(group)
        return {
            'h2h_home_win_rate': home_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_away_win_rate': away_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_draw_rate': draws / total_matches if total_matches > 0 else 0.5
        }
    return df.groupby(['home_team', 'away_team']).apply(calculate_h2h).to_dict()

def get_h2h_history(home_team, away_team, n=5):
    matches = df[(df['home_team'] == home_team) & (df['away_team'] == away_team) |
                 (df['home_team'] == away_team) & (df['away_team'] == home_team)]
    
    # Sort by date and remove duplicates
    matches = matches.sort_values('date', ascending=False).drop_duplicates(subset=['date', 'home_team', 'away_team'])
    
    # Take the most recent n matches
    matches = matches.head(n)
    
    history = []
    for _, match in matches.iterrows():
        date_obj = datetime.strptime(match['date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%d %b %Y')
        
        if match['home_team'] == home_team:
            result = 'home_win' if match['home_goals'] > match['away_goals'] else 'away_win' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'score': f"{match['home_goals']}-{match['away_goals']}",
                'result': result
            })
        else:
            result = 'away_win' if match['home_goals'] > match['away_goals'] else 'home_win' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'home_team': match['away_team'],
                'away_team': match['home_team'],
                'score': f"{match['away_goals']}-{match['home_goals']}",
                'result': result
            })
    return history

def get_team_last_matches(team, n=5):
    home_matches = df[df['home_team'] == team].sort_values('date', ascending=False).head(n)
    away_matches = df[df['away_team'] == team].sort_values('date', ascending=False).head(n)
    
    all_matches = pd.concat([home_matches, away_matches]).sort_values('date', ascending=False).head(n)
    
    history = []
    for _, match in all_matches.iterrows():
        date_obj = datetime.strptime(match['date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%d %b %Y')
        
        if match['home_team'] == team:
            result = 'win' if match['home_goals'] > match['away_goals'] else 'loss' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'opponent': match['away_team'],
                'score': f"{match['home_goals']}-{match['away_goals']}",
                'result': result
            })
        else:
            result = 'win' if match['away_goals'] > match['home_goals'] else 'loss' if match['away_goals'] < match['home_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'opponent': match['home_team'],
                'score': f"{match['away_goals']}-{match['home_goals']}",
                'result': result
            })
    return history

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    
    home_strength, away_strength, home_defense, away_defense = get_team_stats()
    h2h_stats = get_h2h_stats()
    
    try:
        # Calculate features
        home_team_strength = home_strength.get(home_team, np.mean(list(home_strength.values())))
        away_team_strength = away_strength.get(away_team, np.mean(list(away_strength.values())))
        home_team_defense = home_defense.get(home_team, np.mean(list(home_defense.values())))
        away_team_defense = away_defense.get(away_team, np.mean(list(away_defense.values())))
        
        # Use average xG values
        avg_home_xG = df[df['home_team'] == home_team]['home_xG'].mean()
        avg_away_xG = df[df['away_team'] == away_team]['away_xG'].mean()
        
        xG_difference = avg_home_xG - avg_away_xG
        
        # Get head-to-head stats
        h2h = h2h_stats.get((home_team, away_team), {
            'h2h_home_win_rate': 0.5,
            'h2h_away_win_rate': 0.5,
            'h2h_draw_rate': 0.5
        })
        
        # Calculate form
        home_matches = df[df['home_team'] == home_team].tail(5)
        away_matches = df[df['away_team'] == away_team].tail(5)
        
        home_form = (home_matches['home_goals'] - home_matches['away_goals']).mean()
        away_form = (away_matches['away_goals'] - away_matches['home_goals']).mean()
        
        # Prepare the input data
        input_data = pd.DataFrame({
            'home_team_strength': [home_team_strength],
            'away_team_strength': [away_team_strength],
            'home_team_defense': [home_team_defense],
            'away_team_defense': [away_team_defense],
            'home_xG': [avg_home_xG],
            'away_xG': [avg_away_xG],
            'xG_difference': [xG_difference],
            'home_form': [home_form],
            'away_form': [away_form],
            'h2h_home_win_rate': [h2h['h2h_home_win_rate']],
            'h2h_away_win_rate': [h2h['h2h_away_win_rate']],
            'h2h_draw_rate': [h2h['h2h_draw_rate']]
        })
        
        # Make prediction
        prediction_encoded = ensemble_model.predict(input_data)
        prediction = le.inverse_transform(prediction_encoded)[0]
        
        probabilities = ensemble_model.predict_proba(input_data)[0]
        probability_dict = {le.classes_[i]: float(prob) * 100 for i, prob in enumerate(probabilities)}

        # Get head-to-head history
        h2h_history = get_h2h_history(home_team, away_team)
        
        # Get last 5 matches for each team
        home_team_last_matches = get_team_last_matches(home_team)
        away_team_last_matches = get_team_last_matches(away_team)
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probability_dict,
            'h2h_history': h2h_history,
            'home_team_last_matches': home_team_last_matches,
            'away_team_last_matches': away_team_last_matches
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/teams', methods=['GET'])
def get_teams():
    teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    return jsonify(teams)

if __name__ == '__main__':
    app.run(debug=True)