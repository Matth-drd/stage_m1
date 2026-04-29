import pandas as pd
import numpy as np

# Remarque : Dans le code, T représente les équipes T = H ou A

nb_match = 5  # nb match pour calculer la forme récente de l'équipe
weight_g = 1.5  # Poids pour les buts dans l'indicateur de précision pondéré
elo_init = 1500  # score elo avant le premier match pour toutes les équipes

df_init = pd.read_csv("../data/csv/foot_clean.csv")
df = df_init.copy()  # copie du df initial pour éviter de modifier l'original si besoin
df['Date'] = pd.to_datetime(df['Date'])

# Tri chronologique initial pour garantir la cohérence des calculs itératifs
df = df.sort_values(by='Date').reset_index(drop=True)

# COLONNES FULL TIME (FT)


########### FT_ : Tforme / Tatt / Tdef / Tatt_sais / Tdef_sais ############
################## HOME

"""
plus Tdef est faible plus l'équipe T a une défense solide.
Apparition de Nan dans les colonnes créées. Je traite cela à la fin du code.
Les Nan apparaissent pour le 1er match de chaque équipe car les calculs sont effectués
sur les matchs précédents en excluant le match en cours.
"""

df["FT_Hforme"] = df.groupby('HomeTeam')['FTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Hatt"] = df.groupby('HomeTeam')['HS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(level=0,
                                                                                                                drop=True)
df["FT_Hdef"] = df.groupby('HomeTeam')['AS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(level=0,
                                                                                                                drop=True)

avg_h_g_league = df.groupby(["League", "Season"])['FTHG'].transform('mean')
df['FT_Hatt_sais'] = df['FT_Hforme'] / (avg_h_g_league + 0.001)

h_conceded_g = df.groupby('HomeTeam')['FTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
avg_h_e_league = df.groupby(["League", "Season"])['FTAG'].transform('mean')
df['FT_Hdef_sais'] = h_conceded_g / (avg_h_e_league + 0.001)

################## AWAY

df["FT_Aforme"] = df.groupby('AwayTeam')['FTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Aatt"] = df.groupby('AwayTeam')['AS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(level=0,
                                                                                                                drop=True)
df["FT_Adef"] = df.groupby('AwayTeam')['HS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(level=0,
                                                                                                                drop=True)

avg_a_g_league = df.groupby(["League", "Season"])['FTAG'].transform('mean')
df['FT_Aatt_sais'] = df['FT_Aforme'] / (avg_a_g_league + 0.001)

a_conceded_g = df.groupby('AwayTeam')['FTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
avg_a_e_league = df.groupby(["League", "Season"])['FTHG'].transform('mean')
df['FT_Adef_sais'] = a_conceded_g / (avg_a_e_league + 0.001)

print("FT features calculées ✓")

# COLONNES HALF TIME (HT)
########### HT_ : Tforme / Tatt / Tdef / Tatt_sais / Tdef_sais ############
"""
Même logique que pour les stats FT mais basée sur les buts et résultats à la mi-temps.
HT_Hforme  : moyenne des buts marqués à domicile à la mi-temps sur les N derniers matchs
HT_Hatt    : moyenne des tirs tentés à domicile (même colonne HS, les tirs sont sur 90 min)
HT_Hdef    : buts concédés à domicile à la mi-temps (proxy défensif)
HT_Hatt_sais / HT_Hdef_sais : normalisés par la moyenne de la ligue sur la saison

Note : les colonnes HS/AS (tirs) n'ont pas d'équivalent mi-temps dans le dataset.
On utilise donc HTHG/HTAG comme indicateurs offensifs/défensifs mi-temps.
"""

################## HOME — HALF TIME

df["HT_Hforme"] = df.groupby('HomeTeam')['HTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)

# Buts concédés à domicile à la mi-temps -> indicateur défensif
ht_h_conceded = df.groupby('HomeTeam')['HTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["HT_Hdef"] = ht_h_conceded

# Normalisation par saison/ligue
avg_ht_h_g_league = df.groupby(["League", "Season"])['HTHG'].transform('mean')
df['HT_Hatt_sais'] = df['HT_Hforme'] / (avg_ht_h_g_league + 0.001)

avg_ht_h_e_league = df.groupby(["League", "Season"])['HTAG'].transform('mean')
df['HT_Hdef_sais'] = ht_h_conceded / (avg_ht_h_e_league + 0.001)

################## AWAY — HALF TIME

df["HT_Aforme"] = df.groupby('AwayTeam')['HTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)

# Buts concédés à l'extérieur à la mi-temps → indicateur défensif
ht_a_conceded = df.groupby('AwayTeam')['HTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["HT_Adef"] = ht_a_conceded

# Normalisation par saison/ligue
avg_ht_a_g_league = df.groupby(["League", "Season"])['HTAG'].transform('mean')
df['HT_Aatt_sais'] = df['HT_Aforme'] / (avg_ht_a_g_league + 0.001)

avg_ht_a_e_league = df.groupby(["League", "Season"])['HTHG'].transform('mean')
df['HT_Adef_sais'] = ht_a_conceded / (avg_ht_a_e_league + 0.001)

print("HT features calculées ✓")

# JOURS DE REPOS
"""
Le repos est plafonné à 25 jours afin d'éviter les biais liés aux relégations et promotions.
"""

all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
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
            list_repos.append(30)  # Valeur par défaut pour le 1er match

    last_match_date[h_team] = date
    last_match_date[a_team] = date

df['HRepos'], df['ARepos'] = h_repos, a_repos

print("Repos calculé ✓")

# INDICATEURS DE PRÉCISION (FT uniquement — tirs non dispo en HT)
"""
Indicateurs de précision : nb_tir_cadré / nb_tir.
"""
hst_rolling = df.groupby('HomeTeam')['HST'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Hprecision"] = hst_rolling / df["FT_Hatt"].replace(0, np.nan)

ast_rolling = df.groupby('AwayTeam')['AST'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Aprecision"] = ast_rolling / df["FT_Aatt"].replace(0, np.nan)

# Précision pondérée par les buts
df["FT_Hshot"] = df.HST - df.FTHG + df.FTHG * weight_g
Hshot_roll = df.groupby('HomeTeam')["FT_Hshot"].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Hprec_weight"] = Hshot_roll / df["FT_Hatt"].replace(0, np.nan)

df["FT_Ashot"] = df.AST - df.FTAG + df.FTAG * weight_g
Ashot_roll = df.groupby('AwayTeam')["FT_Ashot"].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Aprec_weight"] = Ashot_roll / df["FT_Aatt"].replace(0, np.nan)

print("Précision FT calculée ✓")

# SCORE ELO — FULL TIME
"""
elo_init = 1500
Rappel sur FTR : Domicile=1, Extérieur=2, Nul=0.
https://www.eloratings.net/about
"""

elo_teams = {team: 1500 for team in all_teams}
K = 30

h_elo_before, a_elo_before = [], []

for index, row in df.iterrows():
    H = row['HomeTeam']
    A = row['AwayTeam']
    res = row['FTR']
    ecart_buts = abs(row['FTHG'] - row['FTAG'])

    h_elo_before.append(elo_teams[H])
    a_elo_before.append(elo_teams[A])

    dr = (elo_teams[H] + 100) - elo_teams[A]
    We_H = 1 / (10 ** (-dr / 400) + 1)

    if res == 1:
        W_H, W_A = 1, 0
    elif res == 2:
        W_H, W_A = 0, 1
    else:
        W_H, W_A = 0.5, 0.5

    if ecart_buts <= 1:
        G = 1
    elif ecart_buts == 2:
        G = 1 + 1 / 2
    elif ecart_buts == 3:
        G = 1 + 3 / 4
    else:
        G = 1 + 3 / 4 + (ecart_buts - 3) / 8

    update = K * G * (W_H - We_H)
    elo_teams[H] += update
    elo_teams[A] -= update

df['FT_Elo_H'] = h_elo_before
df['FT_Elo_A'] = a_elo_before
df["FT_Elo_dif"] = df.FT_Elo_H - df.FT_Elo_A

# SCORE ELO — HALFTIME
"""
Même logique que l'ELO FT mais basé sur les résultats à la mi-temps (HTR, HTHG, HTAG).
Permet de capturer la tendance d'une équipe à bien (ou mal) démarrer les matchs.
L'ELO HT est indépendant de l'ELO FT — chaque système reflète une dimension différente.
"""

elo_ht_teams = {team: 1500 for team in all_teams}

h_elo_ht_before, a_elo_ht_before = [], []

for index, row in df.iterrows():
    H = row['HomeTeam']
    A = row['AwayTeam']
    res_ht = row['HTR']
    ecart_buts_ht = abs(row['HTHG'] - row['HTAG'])

    h_elo_ht_before.append(elo_ht_teams[H])
    a_elo_ht_before.append(elo_ht_teams[A])

    dr_ht = (elo_ht_teams[H] + 100) - elo_ht_teams[A]
    We_H_ht = 1 / (10 ** (-dr_ht / 400) + 1)

    if res_ht == 1:
        W_H_ht, W_A_ht = 1, 0
    elif res_ht == 2:
        W_H_ht, W_A_ht = 0, 1
    else:
        W_H_ht, W_A_ht = 0.5, 0.5

    if ecart_buts_ht <= 1:
        G_ht = 1
    elif ecart_buts_ht == 2:
        G_ht = 1 + 1 / 2
    elif ecart_buts_ht == 3:
        G_ht = 1 + 3 / 4
    else:
        G_ht = 1 + 3 / 4 + (ecart_buts_ht - 3) / 8

    update_ht = K * G_ht * (W_H_ht - We_H_ht)
    elo_ht_teams[H] += update_ht
    elo_ht_teams[A] -= update_ht

df['HT_Elo_H'] = h_elo_ht_before
df['HT_Elo_A'] = a_elo_ht_before
df["HT_Elo_dif"] = df.HT_Elo_H - df.HT_Elo_A

print("ELO FT & HT calculés ✓")

# NETTOYAGE DES NaN
print("\nNaN avant nettoyage :")
print(df.isna().sum()[df.isna().sum() > 0])

"""
Les NaN apparaissent pour le 1er match de chaque équipe dans les colonnes rolling.
On supprime ces lignes — elles représentent ~0.9% du dataframe.
"""

L_col = [
    # Full Time
    "FT_Hforme", "FT_Hatt", "FT_Hdef", "FT_Hatt_sais", "FT_Hdef_sais",
    "FT_Aforme", "FT_Aatt", "FT_Adef", "FT_Aatt_sais", "FT_Adef_sais",
    "FT_Hprecision", "FT_Aprecision", "FT_Hprec_weight", "FT_Aprec_weight",
    # Half Time
    "HT_Hforme", "HT_Hdef", "HT_Hatt_sais", "HT_Hdef_sais",
    "HT_Aforme", "HT_Adef", "HT_Aatt_sais", "HT_Adef_sais",
]

df.dropna(axis=0, how='any', subset=L_col, inplace=True)

print("NaN restants :", df.isna().sum().sum())

# Nettoyage des colonnes intermédiaires
df.drop(columns=["FT_Hshot", "FT_Ashot"], inplace=True)

df.to_csv("../data/csv/foot_v3.csv", index=False)
print("CSV sauvegardé ✓")

# RÉCAPITULATIF DES FEATURES

features_FT = ["FT_Hforme", "FT_Hatt", "FT_Hdef", "FT_Aforme", "FT_Aatt", "FT_Adef",
               "FT_Hprecision", "FT_Aprecision", "FT_Hprec_weight", "FT_Aprec_weight",
               "FT_Elo_H", "FT_Elo_A", "FT_Elo_dif"]
# Data leak :
# leak=["FT_Hatt_sais", "FT_Hdef_sais", "FT_Aatt_sais", "FT_Adef_sais"]


features_HT = ["HT_Hforme", "HT_Hdef", "HT_Aforme", "HT_Adef",
               "HT_Elo_H", "HT_Elo_A", "HT_Elo_dif"]
# Data leak :
# leak=["HT_Hatt_sais", "HT_Hdef_sais", "HT_Aatt_sais", "HT_Adef_sais"]

print(f"\nFeatures FT disponibles ({len(features_FT)}) :", features_FT)
print(f"\nFeatures HT disponibles ({len(features_HT)}) :", features_HT)

# On crée les colonnes binaires (1 si vrai, 0 si faux)
df["Hvs"] = (df["FTR"] == 1).astype(int)  # Home vs All
df["Avs"] = (df["FTR"] == 2).astype(int)  # Away vs All
df["Dvs"] = (df["FTR"] == 0).astype(int)  # Draw vs All

df.to_csv("../data/csv/foot_v4.csv", index=False)
print("CSV sauvegardé ✓")