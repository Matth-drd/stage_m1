import pandas as pd
import numpy as np

# Remarque : Dans le code, T représente les équipes T = H ou A

T = 5  # nb match pour calculer la forme récente de l'équipe
weight_g = 1.5  # Poids pour les buts dans l'indicateur de précision pondéré
elo_init = 1500  # score elo avant le premier match pour toutes les équipes

df_init = pd.read_csv("../data/csv/foot_clean.csv")
df = df_init.copy()  # copie du df initial pour éviter de modifier l'original si besoin
df['Date'] = pd.to_datetime(df['Date'])

# Tri chronologique initial pour garantir la cohérence des calculs itératifs
df = df.sort_values(by='Date').reset_index(drop=True)

########### Colonne :  Tforme /      Tatt      /        Tdef       /       Tatt_sais       /       Tdef_sais      ############
########### forme récente /force d'attaque / force de défense / Attaque sur la saison / défense sur la saison  ############
################## HOME
""" 
plus Tdef est faible plus l'équipe T a une défense solide
"""

df["Hforme"] = df.groupby('HomeTeam')['FTHG'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Hatt"] = df.groupby('HomeTeam')['HS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Hdef"] = df.groupby('HomeTeam')['AS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)

# On normalise les données par saisons afin d'avoir la force d'attaque d'une équipe
# par rapport à la force d'attaque des équipes sur toute la saison.
avg_h_g_league = df.groupby(["League", "Season"])['FTHG'].transform('mean')
df['Hatt_sais'] = df['Hforme'] / (avg_h_g_league + 0.001)

h_conceded_fr = df.groupby('HomeTeam')['FTAG'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
avg_h_e_league = df.groupby(["League", "Season"])['FTAG'].transform('mean')
df['Hdef_sais'] = h_conceded_fr / (avg_h_e_league + 0.001)

print(df.head(5))

######################### AWAY
"""
Calcul des statistiques pour l'équipe à l'extérieur.
Note : Adef utilise bien AwayTeam et les tirs de l'équipe à domicile (HS).
"""

df["Aforme"] = df.groupby('AwayTeam')['FTAG'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Aatt"] = df.groupby('AwayTeam')['AS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Adef"] = df.groupby('AwayTeam')['HS'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)

# Normalisation par saisons
avg_a_g_league = df.groupby(["League", "Season"])['FTAG'].transform('mean')
df['Aatt_sais'] = df['Aforme'] / (
        avg_a_g_league + 0.001)  # forme récente de l'équipe / moyenne de buts marqués à l'extérieur dans la ligue

a_conceded_fr = df.groupby('AwayTeam')['FTHG'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
avg_a_e_league = df.groupby(["League", "Season"])['FTHG'].transform('mean')
df['Adef_sais'] = a_conceded_fr / (avg_a_e_league + 0.001)

print(df.head(5))

############ Jours de repos entre 2 matchs colonne : TRepos ############
######################  Jour de repos
"""
Le repos est plafonné à 25 jours afin d’éviter les biais liés aux relégations et promotions. 
Lors des premiers calculs, nous avons remarqué que certaines équipes affichaient jusqu'à 2 600 jours de repos. 
Après une rapide recherche, il s’est avéré que l'équipe en question avait été reléguée en 
division inférieure avant d'être promue à nouveau plusieurs années plus tard.
"""

all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())  # on récupère les listes des équipes H et T
last_match_date = {}

h_repos, a_repos = [], []

for index, row in df.iterrows():
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    date = row['Date']

    for team, list_repos in zip([h_team, a_team], [h_repos, a_repos]):
        if team in last_match_date:
            diff = (date - last_match_date[team]).days
            list_repos.append(min(diff, 25))
        else:
            list_repos.append(15)  # Valeur neutre par défaut

    last_match_date[h_team] = date
    last_match_date[a_team] = date

df['HRepos'], df['ARepos'] = h_repos, a_repos

###############  indicateur de précision sur les 5 derniers matchs  ##############
"""
Indicateurs de précision : nb_tir_cadré / nb_tir.
"""

hst_rolling = df.groupby('HomeTeam')['HST'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Hprecision"] = hst_rolling / df["Hatt"].replace(0, np.nan)

ast_rolling = df.groupby('AwayTeam')['AST'].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Aprecision"] = ast_rolling / df["Aatt"].replace(0, np.nan)

"""
Indicateur de précision avec pondération des buts 
On ajoute 2 colonne, Tprec_weight pour prendre en compte la
précision en pondérant les buts de l'équipe T = H ou A
"""
df["Hshot"] = df.HST - df.FTHG + df.FTHG * weight_g
Hshot_roll = df.groupby('HomeTeam')["Hshot"].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Hprec_weight"] = Hshot_roll / df["Hatt"].replace(0, np.nan)

df["Ashot"] = df.AST - df.FTAG + df.FTAG * weight_g
Hshot_roll = df.groupby('AwayTeam')["Ashot"].rolling(T, min_periods=1).mean().reset_index(level=0, drop=True)
df["Aprec_weight"] = Hshot_roll / df["Aatt"].replace(0, np.nan)

print(df.head())

############ Score ELO ############
"""
elo_init = 1500
Rappel sur FTR : Domicile=1, Extérieur=2, Nul=0.
"""
outcome = [1, .5, 0]  # dans l'ordre victoir de Home, Nul, défaite de Home

# Rn = Ro + K × (W - We)
# W = [1,.5,0] # dans l'ordre victoir de Home, Nul, défaite de Home
# We = 1 / (10(-dr/400) + 1)
# K is then adjusted for the goal difference in the game. It is increased by half if a game is won by two goals,
# by 3/4 if a game is won by three goals, and by 3/4 + (N-3)/8 if the game is won by four or more goals, where N is the goal difference.
# dr equals the difference in ratings plus 100 points for a team playing at home.

# --- Initialisation ---
elo_teams = {team: 1500 for team in all_teams}
K = 30

# Listes pour stocker les résultats
h_elo_before = []
a_elo_before = []

for index, row in df.iterrows():
    H = row['HomeTeam']
    A = row['AwayTeam']
    res = row['FTR']
    ecart_buts = abs(row['FTHG'] - row['FTAG'])

    # 1 STOCKE LES SCORES AVANT LE MATCH
    h_elo_before.append(elo_teams[H])
    a_elo_before.append(elo_teams[A])

    # 2 Calcul du dr (différence avec bonus domicile de 100)
    dr = (elo_teams[H] + 100) - elo_teams[A]

    # 3 Calcul de l'espérance We
    We_H = 1 / (10 ** (-dr / 400) + 1)

    # 4 Déterminer W (résultat réel)
    if res == 1:  # Victoire Home
        W_H, W_A = 1, 0
    elif res == 2:  # Victoire Away
        W_H, W_A = 0, 1
    else:  # Nul
        W_H, W_A = 0.5, 0.5

    # 5 Calculer le multiplicateur d'écart de buts (G)
    if ecart_buts <= 1:
        G = 1
    elif ecart_buts == 2:
        G = 1.5
    elif ecart_buts == 3:
        G = 1.75
    else:
        G = 1.75 + (ecart_buts - 3) / 8

    # 6 Mise à jour du dictionnaire (pour le PROCHAIN match de ces équipes)
    update = K * G * (W_H - We_H)
    elo_teams[H] += update
    elo_teams[A] -= update  # Ce que l'un gagne, l'autre le perd

df['Elo_H_before'] = h_elo_before
df['Elo_A_before'] = a_elo_before

# On peut aprésent créer une colone elo_dif = elo_home - elo_away
df["Elo_dif"] = df.Elo_H_before - df.Elo_A_before

df.to_csv("../data/csv/foot_v3.csv", index=False)
print('CSV sauvegardé')
