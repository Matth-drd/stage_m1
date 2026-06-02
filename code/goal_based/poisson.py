import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss
from tqdm import tqdm

sys.path.append(os.path.abspath('code'))
import config as conf

df = pd.read_csv(conf.path_clean_cell)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by="Date").reset_index(drop=True)
print(f"Nombre de matchs chargés : {len(df)}")

# %% =========================================================================
#  ENTRAÎNEMENT DU MODÈLE DE BASE : ROLLING MAHER (POISSON INDÉPENDANT)
# =========================================================================
toutes_equipes = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

# Initialisation des compteurs historiques
goals_marques_dom = {team: 0 for team in toutes_equipes}
goals_encaisses_dom = {team: 0 for team in toutes_equipes}
goals_marques_ext = {team: 0 for team in toutes_equipes}
goals_encaisses_ext = {team: 0 for team in toutes_equipes}
matches_joues_dom = {team: 0 for team in toutes_equipes}
matches_joues_ext = {team: 0 for team in toutes_equipes}

total_buts_dom, total_buts_ext, total_matchs_ligue = 0, 0, 0

home_teams, away_teams = df["HomeTeam"].values, df["AwayTeam"].values
fthg_goals, ftag_goals = df["FTHG"].values, df["FTAG"].values

lambda_dom_list, lambda_ext_list = [], []
K = 5  # Lissage bayésien

for idx in tqdm(range(len(df)), desc="Calcul des Lambdas"):
    h, a = home_teams[idx], away_teams[idx]

    if total_matchs_ligue == 0:
        moy_ligue_dom, moy_ligue_ext = 1.5, 1.0
    else:
        moy_ligue_dom = total_buts_dom / total_matchs_ligue
        moy_ligue_ext = total_buts_ext / total_matchs_ligue

    # Facteurs d'attaque et de défense
    alpha_h = ((goals_marques_dom[h] + K * moy_ligue_dom) / (matches_joues_dom[h] + K)) / moy_ligue_dom
    delta_h = ((goals_encaisses_dom[h] + K * moy_ligue_ext) / (matches_joues_dom[h] + K)) / moy_ligue_ext
    beta_a = ((goals_encaisses_ext[a] + K * moy_ligue_dom) / (matches_joues_ext[a] + K)) / moy_ligue_dom
    gamma_a = ((goals_marques_ext[a] + K * moy_ligue_ext) / (matches_joues_ext[a] + K)) / moy_ligue_ext

    # Génération des paramètres mu et nu attendus
    lambda_dom_list.append(alpha_h * beta_a * moy_ligue_dom)
    lambda_ext_list.append(gamma_a * delta_h * moy_ligue_ext)

    # Mise à jour des structures de données
    goals_marques_dom[h] += fthg_goals[idx]
    goals_encaisses_dom[h] += ftag_goals[idx]
    goals_marques_ext[a] += ftag_goals[idx]
    goals_encaisses_ext[a] += fthg_goals[idx]
    matches_joues_dom[h] += 1
    matches_joues_ext[a] += 1
    total_buts_dom += fthg_goals[idx]
    total_buts_ext += ftag_goals[idx]
    total_matchs_ligue += 1

df["mu_dom"] = lambda_dom_list
df["nu_ext"] = lambda_ext_list


# %% =========================================================================
#  CONSTRUCTION DU MODÈLE EXTENSION : DIXON-COLES (CORRECTION DE DÉPENDANCE)
# =========================================================================

def log_L_neg_dixon(rho_param, mu_array, nu_array, y_home, y_away):
    rho_val = rho_param[0]
    total_nll = 0.0
    for k in range(len(y_home)):
        x, y = int(y_home[k]), int(y_away[k])
        mu, nu = mu_array[k], nu_array[k]

        # Fonction de correction locale tau(x, y)
        if x == 0 and y == 0:
            tau_val = 1 - mu * nu * rho_val
        elif x == 0 and y == 1:
            tau_val = 1 + mu * rho_val
        elif x == 1 and y == 0:
            tau_val = 1 + nu * rho_val
        elif x == 1 and y == 1:
            tau_val = 1 - rho_val
        else:
            tau_val = 1.0

        if tau_val <= 0:
            return 1e10
        total_nll -= np.log(tau_val)
    return total_nll


# Recherche du rho optimal par maximum de vraisemblance
solution = minimize(
    fun=log_L_neg_dixon,
    x0=[0.0],
    args=(df["mu_dom"].values, df["nu_ext"].values, df["FTHG"].values, df["FTAG"].values),
    method='L-BFGS-B',
    bounds=[(-0.25, 0.10)]
)
rho_optimal = solution.x[0]
print(f"    [OK] Rho optimal Dixon-Coles calculé : {rho_optimal:.4f} (Succès : {solution.success})")

# %% =========================================================================
# INFÉRENCE ET CALCUL DES PROBABILITÉS 1X2 (SÉPARÉES)
# =========================================================================
max_g = 11
buts_possibles = np.arange(max_g)

mu_arr = df["mu_dom"].values
nu_arr = df["nu_ext"].values

# Listes réceptrices pour Maher
p1_m, pX_m, p2_m = [], [], []
# Listes réceptrices pour Dixon-Coles
p1_dc, pX_dc, p2_dc = [], [], []


def tau_correction(x, y, mu, nu, rho):
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    if x == 0 and y == 1:
        return 1 + mu * rho
    if x == 1 and y == 0:
        return 1 + nu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


for idx in range(len(df)):
    mu_val, nu_val = mu_arr[idx], nu_arr[idx]
    p_dom = poisson.pmf(buts_possibles, mu_val)
    p_ext = poisson.pmf(buts_possibles, nu_val)

    # --- CALCULS PROPRES À MAHER ---
    matrix_maher = np.outer(p_dom, p_ext)
    p1_m.append(np.sum(np.tril(matrix_maher, -1)))
    pX_m.append(np.sum(np.diag(matrix_maher)))
    p2_m.append(np.sum(np.triu(matrix_maher, 1)))

    # --- CALCULS PROPRES À DIXON-COLES ---
    matrix_dixon = np.copy(matrix_maher)
    for x in [0, 1]:
        for y in [0, 1]:
            matrix_dixon[x, y] *= tau_correction(x, y, mu_val, nu_val, rho_optimal)

    p1_dc.append(np.sum(np.tril(matrix_dixon, -1)))
    pX_dc.append(np.sum(np.diag(matrix_dixon)))
    p2_dc.append(np.sum(np.triu(matrix_dixon, 1)))

# Injection des résultats dans le DataFrame global
df["p_1_Maher"], df["p_X_Maher"], df["p_2_Maher"] = p1_m, pX_m, p2_m
df["p_1_DC"], df["p_X_DC"], df["p_2_DC"] = p1_dc, pX_dc, p2_dc

# %% =========================================================================
# FILTRAGE ET VALIDATION CHRONOLOGIQUE (WARM-UP)
# =========================================================================
match_nb_home = df.groupby("HomeTeam").cumcount()
match_nb_away = df.groupby("AwayTeam").cumcount()

# Conservation des données lorsque les équipes ont au moins 4 matchs d'historique
df_fiable = df[(match_nb_home >= 4) & (match_nb_away >= 4)].reset_index(drop=True)
print(f"\nMatchs conservés après phase de warm-up : {len(df_fiable)} / {len(df)}")

# %% =========================================================================
# ÉVALUATION ET COMPARAISON DES DEUX MODÈLES
# =========================================================================
# Encodage cibles réelles (0=Victoire Dom, 1=Nul, 2=Victoire Ext)
y_true_1X2 = np.where(df_fiable["FTHG"] > df_fiable["FTAG"], 0,
                      np.where(df_fiable["FTHG"] == df_fiable["FTAG"], 1, 2))
y_true_dom = (y_true_1X2 == 0).astype(int)

# Récupération des blocs de probabilités
probs_maher = df_fiable[["p_1_Maher", "p_X_Maher", "p_2_Maher"]].values
probs_dixon = df_fiable[["p_1_DC", "p_X_DC", "p_2_DC"]].values

# --- CALCUL DES MÉTRIQUES ---
metrics_summary = {
    "Maher": {
        "Log-Loss 1X2": log_loss(y_true_1X2, probs_maher),
        "Brier Score (H)": brier_score_loss(y_true_dom, probs_maher[:, 0])
    },
    "Dixon-Coles": {
        "Log-Loss 1X2": log_loss(y_true_1X2, probs_dixon),
        "Brier Score (H)": brier_score_loss(y_true_dom, probs_dixon[:, 0])
    }
}

print("\n" + "=" * 65)
print("       COMPARAISON STATISTIQUE : MAHER vs DIXON & COLES")
print("=" * 65)
print(f"Log-Loss 1X2 (Maher)        : {metrics_summary['Maher']['Log-Loss 1X2']:.4f}")
print(f"Log-Loss 1X2 (Dixon-Coles)  : {metrics_summary['Dixon-Coles']['Log-Loss 1X2']:.4f}")
print("-" * 65)
print(f"Brier Score Home (Maher)    : {metrics_summary['Maher']['Brier Score (H)']:.4f}")
print(f"Brier Score Home (Dixon)    : {metrics_summary['Dixon-Coles']['Brier Score (H)']:.4f}")
print("=" * 65)

# %% =========================================================================
# ÉVALUATION DES PRÉDICTIONS DE BUTS
# =========================================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance

buts_réels_dom = df_fiable["FTHG"].values
buts_réels_ext = df_fiable["FTAG"].values
mu_attendus = df_fiable["mu_dom"].values
nu_attendus = df_fiable["nu_ext"].values

# 1. Calcul des trois métriques pour l'équipe à Domicile
mae_dom = mean_absolute_error(buts_réels_dom, mu_attendus)
mse_dom = mean_squared_error(buts_réels_dom, mu_attendus)
dev_dom = mean_poisson_deviance(buts_réels_dom, mu_attendus)

# 2. Calcul des trois métriques pour l'équipe à l'Extérieur
mae_ext = mean_absolute_error(buts_réels_ext, nu_attendus)
mse_ext = mean_squared_error(buts_réels_ext, nu_attendus)
dev_ext = mean_poisson_deviance(buts_réels_ext, nu_attendus)

print("--- EQUIPE A DOMICILE (Paramètre Mu) ---")
print(f"Déviance de Poisson : {dev_dom:.4f}  (Qualité de la loi de distribution)")
print(f"MSE (Erreur carrée) : {mse_dom:.4f}  (Pénalise fortement les gros écarts)")
print(f"MAE (Erreur absolue): {mae_dom:.4f} buts d'écart moyen")
print("-" * 65)
print("--- EQUIPE A L'EXTÉRIEUR (Paramètre Nu) ---")
print(f"Déviance de Poisson : {dev_ext:.4f}")
print(f"MSE (Erreur carrée) : {mse_ext:.4f}")
print(f"MAE (Erreur absolue): {mae_ext:.4f} buts d'écart moyen")
print("=" * 65)


# %% =========================================================================
# BACKTEST ET PERFORMANCE FINANCIÈRE (VALUE BETS)
# =========================================================================
def simuler_performance_financiere(y_true, probs_modele, cotes, nom_modele, mise=10.0):
    implied_prob = 1 / cotes
    preds_decision = (probs_modele > implied_prob).astype(int)

    mask = (preds_decision == 1)
    total_investi = mask.sum() * mise

    if total_investi == 0:
        return nom_modele, 0, 0.0, 0.0

    gains = np.where(y_true[mask] == 1, mise * cotes[mask], 0.0)
    profit_net = gains.sum() - total_investi
    roi = (profit_net / total_investi) * 100

    return nom_modele, mask.sum(), total_investi, profit_net, roi


# Extraction des cotes du bookmaker
cotes_home = df_fiable.get("B365H", df_fiable.get("AvgH", 2.0)).values

# Exécution des simulations indépendantes
res_M = simuler_performance_financiere(y_true_dom, df_fiable["p_1_Maher"].values, cotes_home, "MAHER")
res_D = simuler_performance_financiere(y_true_dom, df_fiable["p_1_DC"].values, cotes_home, "DIXON-COLES")

print("\n" + "=" * 65)
print(f"       SIMULATION FINANCIÈRE : STRATÉGIE VALUE BET (HOME)")
print("=" * 65)
print(f"[{res_M[0]}] Pari(s): {res_M[1]} | Investi: {res_M[2]}€ | Profit: {res_M[3]:.2f}€ | ROI: {res_M[4]:.2f}%")
print(f"[{res_D[0]}] Pari(s): {res_D[1]} | Investi: {res_D[2]}€ | Profit: {res_D[3]:.2f}€ | ROI: {res_D[4]:.2f}%")
print("=" * 65)




# %%
print("Dixon ajouter fonction pondération temporelle")