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
    
    leagues = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']
    
    for year in range(start_year, end_year + 1):
        for league in leagues:
            print(f"Scraping data for {league} {year}")
            data = get_data(league, year)
            
            data['league'] = league
            data['season'] = year
            
            all_data.append(data)
            
            time.sleep(2)  # Be nice to the server
    
    return pd.concat(all_data, ignore_index=True)

# Usage
df = scrape_football_data(2014, 2023)

# Data cleaning and transformation
df['date'] = pd.to_datetime(df['datetime']).dt.date
df['home_team'] = df['h'].apply(lambda x: x['title'])
df['away_team'] = df['a'].apply(lambda x: x['title'])
df['home_score'] = df['goals'].apply(lambda x: x['h'])
df['away_score'] = df['goals'].apply(lambda x: x['a'])

# Select and reorder columns
columns_order = ['date', 'league', 'season', 'home_team', 'away_team', 'home_score', 'away_score']
df = df[columns_order]

# Save to CSV
df.to_csv('football_data.csv', index=False)
print(f"Data scraped and saved to football_data.csv. Total matches: {len(df)}")

# Print first few rows of the processed data
print(df.head())