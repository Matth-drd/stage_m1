# %%
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('code'))
import config as conf
import numpy as np
from scipy.stats import poisson
from tqdm import tqdm

# %%
df = pd.read_csv(conf.path_clean_cell)
print(f"Nombre de matchs chargés : {len(df)}")

# %% =========================================================================
# CALCUL DES LAMBDAS (MU_DOM, NU_EXT) — MODÈLE DE MAHER ADAPTÉ (ROLLING + SMOOTHING)
# =========================================================================

# Initialisation des structures
toutes_equipes = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

goals_marques_dom = {team: 0 for team in toutes_equipes}
goals_encaisses_dom = {team: 0 for team in toutes_equipes}
goals_marques_ext = {team: 0 for team in toutes_equipes}
goals_encaisses_ext = {team: 0 for team in toutes_equipes}

matches_joues_dom = {team: 0 for team in toutes_equipes}
matches_joues_ext = {team: 0 for team in toutes_equipes}

# Compteurs glissants de la ligue pour éviter tout data leakage
total_buts_dom = 0
total_buts_ext = 0
total_matchs_ligue = 0

# Vectorisation pour optimiser la vitesse d'exécution
home_teams = df["HomeTeam"].values
away_teams = df["AwayTeam"].values
fthg_goals = df["FTHG"].values
ftag_goals = df["FTAG"].values

# Listes pour stocker les lambdas et l'historique temporel des coefficients
lambda_dom_list = []
lambda_ext_list = []
alpha_instant_list = []
delta_instant_list = []
beta_instant_list = []
gamma_instant_list = []

K = 5  # Paramètre de lissage bayésien

for idx in tqdm(range(len(df))):
    h = home_teams[idx]
    a = away_teams[idx]

    # Calcul dynamique des moyennes de la ligue à cet instant t
    if total_matchs_ligue == 0:
        moy_ligue_dom = 1.5  # moyenne sur le dataset de but domicile
        moy_ligue_ext = 1.0  # moyenne sur le dataset des buts ext
    else:
        moy_ligue_dom = total_buts_dom / total_matchs_ligue
        moy_ligue_ext = total_buts_ext / total_matchs_ligue

    # Lissage Bayésien + Normalisation par la moyenne de la ligue 
    alpha_h = ((goals_marques_dom[h] + K * moy_ligue_dom) / (matches_joues_dom[h] + K)) / moy_ligue_dom
    delta_h = ((goals_encaisses_dom[h] + K * moy_ligue_ext) / (matches_joues_dom[h] + K)) / moy_ligue_ext

    beta_a = ((goals_encaisses_ext[a] + K * moy_ligue_dom) / (matches_joues_ext[a] + K)) / moy_ligue_dom
    gamma_a = ((goals_marques_ext[a] + K * moy_ligue_ext) / (matches_joues_ext[a] + K)) / moy_ligue_ext

    # Calcul des espérances de Poisson (Lambdas) selon Maher
    mu = alpha_h * beta_a * moy_ligue_dom
    nu = gamma_a * delta_h * moy_ligue_ext

    # Stockage des valeurs calculées
    lambda_dom_list.append(mu)
    lambda_ext_list.append(nu)
    alpha_instant_list.append(alpha_h)
    delta_instant_list.append(delta_h)
    beta_instant_list.append(beta_a)
    gamma_instant_list.append(gamma_a)

    # 4. Mise à jour des compteurs individuels et globaux (Strict Rolling après calcul)
    goals_marques_dom[h] += fthg_goals[idx]
    goals_encaisses_dom[h] += ftag_goals[idx]
    goals_marques_ext[a] += ftag_goals[idx]
    goals_encaisses_ext[a] += fthg_goals[idx]

    matches_joues_dom[h] += 1
    matches_joues_ext[a] += 1

    total_buts_dom += fthg_goals[idx]
    total_buts_ext += ftag_goals[idx]
    total_matchs_ligue += 1

# Assignation des variables au DataFrame une seule fois en sortie de boucle
df["mu_dom"] = lambda_dom_list
df["nu_ext"] = lambda_ext_list

# Sauvegarde des paramètres instantanés pour tes futurs graphiques (Plus de pic de zéros !)
df["alpha_instant"] = alpha_instant_list
df["delta_instant"] = delta_instant_list
df["beta_instant"] = beta_instant_list
df["gamma_instant"] = gamma_instant_list

# %% =========================================================================
# TABLEAU RÉCAPITULATIF DES PARAMÈTRES FINAUX
# =========================================================================
# Note : Pour éviter l'effondrement des valeurs en fin de saison à cause de sqrt(Sx),
# nous utilisons ici la moyenne globale réelle de fin de saison.

final_moy_dom = df["FTHG"].mean()
final_moy_ext = df["FTAG"].mean()

toutes_equipes_triee = sorted(list(toutes_equipes))

stats_maher = {
    "home attack (alpha)": [],
    "away defence (beta)": [],
    "home defence (gamma)": [],
    "away attack (delta)": []
}

for team in toutes_equipes_triee:
    # Calcul des forces finales normalisées par rapport à la moyenne de fin de saison
    alpha_fin = (goals_marques_dom[team] / max(matches_joues_dom[team], 1)) / final_moy_dom
    delta_fin = (goals_encaisses_dom[team] / max(matches_joues_dom[team], 1)) / final_moy_ext
    beta_fin = (goals_encaisses_ext[team] / max(matches_joues_ext[team], 1)) / final_moy_dom
    gamma_fin = (goals_marques_ext[team] / max(matches_joues_ext[team], 1)) / final_moy_ext

    stats_maher["home attack (alpha)"].append(round(alpha_fin, 2))
    stats_maher["away defence (beta)"].append(round(beta_fin, 2))
    stats_maher["home defence (gamma)"].append(round(delta_fin, 2))  # gamma de la table = faiblesse défensive dom
    stats_maher["away attack (delta)"].append(round(gamma_fin, 2))  # delta de la table = attaque ext

df_table_maher = pd.DataFrame(stats_maher, index=toutes_equipes_triee)
df_table_maher.index.name = "Team"

# %% =========================================================================
# CALCUL DES PROBABILITÉS DE BUTS PAR MATCH — LOI DE POISSON
# =========================================================================
max_g = 11
mu_arr = df["mu_dom"].values
nu_arr = df["nu_ext"].values

res = {
    "HomeTeam": df["HomeTeam"].values,
    "AwayTeam": df["AwayTeam"].values,
}

for buts in range(max_g + 1):
    res[f"dom_{buts}_but"] = poisson.pmf(buts, mu_arr)
    res[f"ext_{buts}_but"] = poisson.pmf(buts, nu_arr)

df_recap_global = pd.DataFrame(res)

# %% =========================================================================
# PRÉDICTION DU SCORE FINAL — MATRICE JOINTE P(X,Y) = P(X) × P(Y)
# =========================================================================
results = []

for idx, row in df_recap_global.iterrows():
    p_dom = np.array([row[f"dom_{i}_but"] for i in range(max_g)])
    p_ext = np.array([row[f"ext_{i}_but"] for i in range(max_g)])

    score_matrix = np.outer(p_dom, p_ext)

    proba_score_max = np.max(score_matrix)
    best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

    pred_dom = best_idx[0]
    pred_ext = best_idx[1]

    p_home_win = np.sum(np.tril(score_matrix, -1))
    p_draw = np.sum(np.diag(score_matrix))
    p_away_win = np.sum(np.triu(score_matrix, 1))

    results.append({
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "Score_Predit": f"{pred_dom} - {pred_ext}",
        "Proba_Score": round(proba_score_max, 4),
        "p_home_win": round(p_home_win, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away_win, 4),
    })

df_predictions = pd.DataFrame(results)
df_predictions.dropna(inplace=True)
print("\n--- TABLEAU FINAL DES PRÉDICTIONS ---")
print(df_predictions.head(200))

# %% =========================================================================
# ÉVALUATION DU MODÈLE
# =========================================================================
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance
)

y_home = df["FTHG"].values
y_away = df["FTAG"].values
y_home_pred = df["mu_dom"].values
y_away_pred = df["nu_ext"].values

y_home_pred = np.nan_to_num(y_home_pred, nan=df["FTHG"].mean())
y_away_pred = np.nan_to_num(y_away_pred, nan=df["FTAG"].mean())

y_home_pred = np.clip(y_home_pred, 1e-5, None)
y_away_pred = np.clip(y_away_pred, 1e-5, None)

home_mean = np.nanmean(y_home)
away_mean = np.nanmean(y_away)

y_home_baseline = np.full(len(df), home_mean)
y_away_baseline = np.full(len(df), away_mean)

y_true_global = np.concatenate([y_home, y_away])
y_pred_global = np.concatenate([y_home_pred, y_away_pred])
y_base_global = np.concatenate([y_home_baseline, y_away_baseline])

mask_valid = ~np.isnan(y_true_global)
y_true_global = y_true_global[mask_valid]
y_pred_global = y_pred_global[mask_valid]
y_base_global = y_base_global[mask_valid]

mae_home = mean_absolute_error(y_home, y_home_pred)
mse_home = (mean_squared_error(y_home, y_home_pred))
poisson_home = mean_poisson_deviance(y_home, y_home_pred)

mae_away = mean_absolute_error(y_away, y_away_pred)
mse_away = (mean_squared_error(y_away, y_away_pred))
poisson_away = mean_poisson_deviance(y_away, y_away_pred)

mae_global = mean_absolute_error(y_true_global, y_pred_global)
mse_global = (mean_squared_error(y_true_global, y_pred_global))
poisson_global = mean_poisson_deviance(y_true_global, y_pred_global)

mae_global_base = mean_absolute_error(y_true_global, y_base_global)
mse_global_base = (mean_squared_error(y_true_global, y_base_global))
poisson_global_base = mean_poisson_deviance(y_true_global, y_base_global)

gain_mae = 100 * (mae_global_base - mae_global) / mae_global_base
gain_mse = 100 * (mse_global_base - mse_global) / mse_global_base
gain_poisson = 100 * (poisson_global_base - poisson_global) / poisson_global_base

print("\n" + "=" * 70)
print("MÉTRIQUES DE PERFORMANCE DU MODÈLE DE POISSON")
print("=" * 70)
print(f"MAE domicile           : {mae_home:.4f}")
print(f"mse domicile          : {mse_home:.4f}")
print(f"Poisson Deviance dom.  : {poisson_home:.4f}")
print("-" * 70)
print(f"MAE extérieur          : {mae_away:.4f}")
print(f"mse extérieur         : {mse_away:.4f}")
print(f"Poisson Deviance ext.  : {poisson_away:.4f}")
print("-" * 70)
print(f"MAE global             : {mae_global:.4f}")
print(f"mse global            : {mse_global:.4f}")
print(f"Poisson Deviance glob. : {poisson_global:.4f}")

print("\n" + "=" * 70)
print("AMÉLIORATION DU MODÈLE VS BASELINE")
print("=" * 70)
print(f"Gain MAE               : {gain_mae:.2f}%")
print(f"Gain mse              : {gain_mse:.2f}%")
print(f"Gain Poisson Deviance  : {gain_poisson:.2f}%")
print("=" * 70)
# %%
