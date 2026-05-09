import pandas as pd
import numpy as np

# Remarque : Dans le code, T représente les équipes T = H ou A

nb_match = 5  # nb match pour calculer la forme récente de l'équipe
weight_g = 1.5  # Poids pour les buts dans l'indicateur de précision pondéré
elo_init = 1500  # score elo avant le premier match pour toutes les équipes

df_init = pd.read_csv("../data/csv/foot_.csv")
df = df_init.copy()  # copie du df initial pour éviter de modifier l'original si besoin
df['Date'] = pd.to_datetime(df['Date'])

# Tri chronologique initial pour garantir la cohérence des calculs itératifs
df = df.sort_values(by='Date').reset_index(drop=True)

# COLONNES FULL TIME (FT)


########### FT_ : Tforme / Tatt / Tdef /  ############
################## HOME

"""
plus Tdef est faible plus l'équipe T a une défense solide.
Apparition de Nan dans les colonnes créées. Je traite cela à la fin du code.
Les Nan apparaissent pour le 1er match de chaque équipe car les calculs sont effectués
sur les matchs précédents en excluant le match en cours.
"""

df["FT_Hforme"] = df.groupby('HomeTeam')['FTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Aforme"] = df.groupby('AwayTeam')['FTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)

df["FT_Hatt"] = df.groupby('HomeTeam')['HS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Aatt"] = df.groupby('AwayTeam')['AS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Hdef"] = df.groupby('HomeTeam')['AS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["FT_Adef"] = df.groupby('AwayTeam')['HS'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
print(df.head())
print("FT features calculées  ")
# ==================
# Différences
# ==================
df["FT_forme_diff"] = df.FT_Hforme - df.FT_Aforme
df['FT_att_diff'] = df.FT_Hatt - df.FT_Aatt
df["FT_def_diff"] = df.FT_Hdef - df.FT_Adef

# COLONNES HALF TIME (HT)
########### HT_ : Tforme / Tatt / Tdef /  ############
"""
Même logique que pour les stats FT mais basée sur les buts et résultats à la mi-temps.
HT_Hforme  : moyenne des buts marqués à domicile à la mi-temps sur les N derniers matchs
HT_Hatt    : moyenne des tirs tentés à domicile (même colonne HS, les tirs sont sur 90 min)
HT_Hdef    : buts concédés à domicile à la mi-temps (proxy défensif)


Note : les colonnes HS/AS (tirs) n'ont pas d'équivalent mi-temps dans le dataset.
On utilise donc HTHG/HTAG comme indicateurs offensifs/défensifs mi-temps.
"""

################## HALF TIME

df["HT_Hforme"] = df.groupby('HomeTeam')['HTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["HT_Aforme"] = df.groupby('AwayTeam')['HTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
# Buts concédés à domicile à la mi-temps -> indicateur défensif
df["HT_Hdef"] = df.groupby('HomeTeam')['HTAG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)
df["HT_Adef"] = df.groupby('AwayTeam')['HTHG'].rolling(nb_match, min_periods=1, closed='left').mean().reset_index(
    level=0, drop=True)

# ======================
#  différence
# ======================
df["HT_forme_diff"] = df.HT_Hforme - df.HT_Aforme
df['HT_def_diff'] = df.HT_Hdef - df.HT_Adef
print(df.head())
print("HT features calculées  ")

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

# différence
df["Repos_diff"] = df.HRepos - df.ARepos

print(df.head())
print("Repos calculé ")

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

df["FT_prec_diff"] = df.FT_Hprecision - df.FT_Aprecision
df["FT_prec_weight_diff"] = df.FT_Hprec_weight - df.FT_Aprec_weight
print(df.head())

print("Précision FT calculée  ")

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

print(df.head())
print("ELO FT & HT calculés  ")

# ==================================
# indicateur de fautes
# ==================================
# Fautes (fouls)
df["FT_HavgF"] = (df.groupby('HomeTeam')['HF'].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df["FT_AavgF"] = (df.groupby('AwayTeam')["AF"].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df["FT_avgF_diff"] = df.FT_HavgF - df.FT_AavgF
# Carton jaune (Yellow)
df["FT_HavgY"] = (df.groupby('HomeTeam')['HY'].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df['FT_AavgY'] = (df.groupby('AwayTeam')["AY"].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df['FT_avgY_diff'] = df.FT_HavgY - df.FT_AavgY
# Carton Rouge (Red card)
df["FT_HavgR"] = (df.groupby('HomeTeam')["HR"].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df["FT_AavgR"] = (df.groupby('AwayTeam')["AR"].rolling(nb_match, min_periods=1, closed='left')
                  .mean().reset_index(level=0,
                                      drop=True))
df["FT_avgR_diff"] = df.FT_HavgR - df.FT_AavgR

# df.to_csv("../data/csv/foot_v3.csv", index=False)
# print("CSV sauvegardé  ")


# ========================
# Suppression des colonnes de paris
# ========================
betting_col_to_drop = [
    'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
    'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA']
betting = ['B365H', 'B365D', 'B365A']

df.drop(columns=betting_col_to_drop, inplace=True)
print(df.columns)
#
# Ratio
ratio = [
    ("FT_Hforme", "FT_Aforme", "FT_forme_ratio"),
    ("FT_Hatt", 'FT_Aatt', "FT_att_ratio"),
    ("FT_Hdef", 'FT_Adef', "FT_def_ratio"),
    ("HT_Hforme", 'HT_Aforme', 'HT_forme_ratio'),
    ('HT_Hdef', "HT_Adef", 'HT_def_ratio'),
    ('HRepos', "ARepos", 'Repos_ratio'),
    ("FT_Hprecision", "FT_Aprecision", "FT_prec_ratio"),
    ("FT_Hprec_weight", "FT_Aprec_weight", "FT_prec_weight_ratio"),
    ("FT_Elo_H", "FT_Elo_A", "FT_Elo_ratio"),
    ("HT_Elo_H", "HT_Elo_A", "HT_Elo_ratio"),
    ("FT_HavgF", "FT_AavgF", "FT_avgF_ratio"),
    ("FT_HavgY", "FT_AavgY", "FT_avgY_ratio"),
    ("FT_HavgR", "FT_AavgR", "FT_avgR_ratio")
]
for h, a, r in ratio:
    df[r] = df[h] + 1 / (df[a] + 1)  # pour éviter de divisier par 0
print(df.columns)
print("Ratio")


def streaks(df):
    all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    win_count = {team: 0 for team in all_teams}
    lose_count = {team: 0 for team in all_teams}

    h_win, a_win = [], []
    h_lose, a_lose = [], []

    for idx, row in df.iterrows():
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']

        # --- ÉTAPE 1 : ON RÉCUPÈRE LES STREAKS AVANT LE MATCH ---
        h_win.append(win_count[h_team])
        a_win.append(win_count[a_team])
        h_lose.append(lose_count[h_team])
        a_lose.append(lose_count[a_team])

        # --- ÉTAPE 2 : ON MET À JOUR LES COMPTEURS POUR LA SUITE ---
        res = row['FTR']

        if res == 1:  # Victoire Domicile
            win_count[h_team] += 1
            lose_count[h_team] = 0
            win_count[a_team] = 0
            lose_count[a_team] += 1
        elif res == 2:  # Victoire Extérieur
            win_count[a_team] += 1
            lose_count[a_team] = 0
            win_count[h_team] = 0
            lose_count[h_team] += 1
        else:  # Match Nul (0)
            win_count[h_team] = 0
            lose_count[h_team] = 0
            win_count[a_team] = 0
            lose_count[a_team] = 0

    return h_win, a_win, h_lose, a_lose


df['H_WinStreak'], df['A_WinStreak'], df['H_LoseStreak'], df['A_LoseStreak'] = streaks(df)
df['WinStreak_diff'] = df['H_WinStreak'] - df['A_WinStreak']
df['LoseStreak_diff'] = df['H_LoseStreak'] - df['A_LoseStreak']
print("Série victoire/défaite")

# On crée les colonnes binaires (1 si vrai, 0 si faux)
df["Hvs"] = (df["FTR"] == 1).astype(int)  # Home vs All
df["Avs"] = (df["FTR"] == 2).astype(int)  # Away vs All
df["Dvs"] = (df["FTR"] == 0).astype(int)  # Draw vs All

print(df.info())
print(df.columns)

# ==================================
# ==================================
# NETTOYAGE DES NaN
# ==================================

print("\nNaN avant nettoyage :")
print(df.isna().sum()[df.isna().sum() > 0])

"""
Les NaN apparaissent pour le 1er match de chaque équipe dans les colonnes rolling.
On supprime ces lignes — elles représentent ~0.9% du dataframe.
"""

L_col = [
    # Full Time
    "FT_Hforme", "FT_Hatt", "FT_Hdef", "FT_Aforme", "FT_Aatt", "FT_Adef",
    "FT_Hprecision", "FT_Aprecision", "FT_Hprec_weight", "FT_Aprec_weight",
    # Half Time
    "HT_Hforme", "HT_Hdef",
    "HT_Aforme", "HT_Adef"]

df.dropna(axis=0, how='any', subset=L_col, inplace=True)

print("NaN restants :", df.isna().sum().sum())

# Nettoyage des colonnes intermédiaires
df.drop(columns=["FT_Hshot", "FT_Ashot"], inplace=True)

print(df.info())
print(df.value_counts("FTR"))
# =================
# Sauvegarde du dataframe en CSV
# =================
df.to_csv("../data/csv/foot_v4.csv", index=False)
print("CSV sauvegardé")

# 'Hvs', 'Avs', 'Dvs' permette de comparer 1 résultat spécifique par rapport au reste
