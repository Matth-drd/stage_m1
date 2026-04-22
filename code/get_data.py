import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def fetch_top_5_leagues_data():
    leagues = {
        'E0': 'Premier League', 
        'F1': 'Ligue 1', 
        'D1': 'Bundesliga', 
        'I1': 'Serie A', 
        'SP1': 'La Liga'
    }
    seasons = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324']
    base_url = "https://www.football-data.co.uk/mmz4281/"
    
    all = []
    for season in seasons:
        for league_code, league_name in leagues.items():
            url = f"{base_url}{season}/{league_code}.csv"
            df = pd.read_csv(url, encoding='unicode_escape')
            
            df['Season'] = season
            df['League'] = league_name
            
            all.append(df)
            print(f"Succès : {league_name} - Saison {season}")

    dataset = pd.concat(all, ignore_index=True)
    
    return dataset

if __name__ == "__main__":
    raw_data = fetch_top_5_leagues_data()
    # final_data = clean_football_data(raw_data)

    # print(f"\nDimensions du dataset final : {final_data.shape[0]} matchs, {final_data.shape[1]} variables.")
