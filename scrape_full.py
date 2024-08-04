import requests
import pandas as pd
import json
from datetime import datetime
import time

def get_data(league, year):
    url = f'https://understat.com/league/{league}/{year}'
    
    res = requests.get(url)
    
    # Find the JSON data in the HTML
    json_data = res.text.split("JSON.parse('")[1].split("')")[0]
    
    # Replace escaped characters
    json_data = json_data.encode('utf8').decode('unicode_escape')
    
    # Parse JSON
    data = json.loads(json_data)
    
    return pd.DataFrame(data)

def scrape_football_data(start_year, end_year):
    all_data = []
    
    leagues = {'EPL': 'english_premier_league',
               'La_liga': 'spanish_la_liga',
               'Bundesliga': 'german_bundesliga',
               'Serie_A': 'italian_serie_a',
               'Ligue_1': 'french_ligue_1'}
    
    for year in range(start_year, end_year + 1):
        for league, league_name in leagues.items():
            print(f"Scraping data for {league} {year}")
            data = get_data(league, year)
            
            df = pd.DataFrame(data)
            df['season'] = f'{year}-{year+1}'
            df['league'] = league_name
            
            all_data.append(df)
            
            time.sleep(2)  # Be nice to the server
    
    return pd.concat(all_data, ignore_index=True)

# Usage
df = scrape_football_data(2014, 2024)

# Print column names
print("Available columns:", df.columns.tolist())

# Data cleaning and transformation
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time

df['home_team'] = df['h'].apply(lambda x: json.loads(x)['title'] if isinstance(x, str) else x['title'])
df['away_team'] = df['a'].apply(lambda x: json.loads(x)['title'] if isinstance(x, str) else x['title'])

df['home_goals'] = df['goals'].apply(lambda x: json.loads(x)['h'] if isinstance(x, str) else x['h'])
df['away_goals'] = df['goals'].apply(lambda x: json.loads(x)['a'] if isinstance(x, str) else x['a'])

df['home_xG'] = df['xG'].apply(lambda x: json.loads(x)['h'] if isinstance(x, str) else x['h'])
df['away_xG'] = df['xG'].apply(lambda x: json.loads(x)['a'] if isinstance(x, str) else x['a'])

# Select and reorder columns
columns_to_use = ['date', 'time', 'league', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xG', 'away_xG']
df = df[columns_to_use]

# Convert numeric columns
numeric_columns = ['home_goals', 'away_goals', 'home_xG', 'away_xG']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Save to CSV
df.to_csv('football_match_data.csv', index=False)
print(f"Data scraped and saved to football_match_data.csv. Total matches: {len(df)}")

# Print first few rows of the processed data
print(df.head())

# Print column names of the final dataframe
print("Final columns:", df.columns.tolist())