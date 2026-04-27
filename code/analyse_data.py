import pandas as pd

T = 5  # nb match pour calculer la forme récente de l'équipe

df_init = pd.read_csv("../data/csv/foot_clean.csv")
df = df_init.copy()

################## HOME
# Rq comment calculer la moyenne des 1,2,3 et 4 premier match ?
df.sort_values(by=['HomeTeam', 'Date'], inplace=True)
df["Hfr"] = df.groupby('HomeTeam')['FTHG'].rolling(window=T, min_periods=1).mean().reset_index(level=0, drop=True)  # Création de la colonne Home Forme Recente Hfr
df["Hfa"] = df.groupby('HomeTeam')['HS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
# df["Hfd"] =  # création de Home Force défense

print(df.head(5))
######################### AWAY
df.sort_values(by=['AwayTeam', 'Date'], inplace=True)
df["Afr"] = df.groupby('AwayTeam')['FTAG'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)  # Création de la colonne Away Forme Recente Afr
df["Afa"] = df.groupby('AwayTeam')['AS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
# df["Afd"] =

print(df.head(5))