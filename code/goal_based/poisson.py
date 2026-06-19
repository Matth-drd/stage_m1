"""
=============================================================================
MODÈLES STATISTIQUES POISSON POUR LA PRÉDICTION DE RÉSULTATS
=============================================================================
Comparaison des modèles Maher (Poisson ) et Dixon-Coles (avec correction
des petits scores). Inclut la version temporelle pour améliorer les estimations
récentes et un backtest value betting.

Données : football-data.co.uk (matchs historiques par ligue/saison)
Méthodes : Rolling Maher, Maximum Vraisemblance, test du Chi-2 d'indépendance
Sortie : Calibration des paramètres, métriques ML, simulations financières, graphes.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, chi2_contingency
from sklearn.metrics import (brier_score_loss, log_loss, mean_squared_error,
                             mean_absolute_error, mean_poisson_deviance)
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath('code'))
import config as conf

# =============================================================================
# CHARGEMENT ET PRÉPARATION INITIALE
# =============================================================================

df = pd.read_csv(conf.path_clean_cell)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by="Date").reset_index(drop=True)
print(f"Nombre de matchs chargés : {len(df)}")

# =============================================================================
# TEST DU CHI-2 D'INDÉPENDANCE
# =============================================================================
# Vérifie si les buts marqués par le HOME et l'AWAY sont indépendants.
# Null : les buts sont indépendants (favorable aux modèles séparés Poisson).
# Alternative : dépendance significative (nécessite correction comme Dixon-Coles).

df_chi2 = df.copy()
# On clippe les buts élevés pour améliorer la qualité du test
df_chi2["FTHG"] = df_chi2["FTHG"].clip(upper=3)
df_chi2["FTAG"] = df_chi2["FTAG"].clip(upper=2)
table_contingence = pd.crosstab(
    df_chi2["FTHG"], df_chi2["FTAG"],
    rownames=['Buts Domicile'],
    colnames=['Buts Extérieur'])

chi2, p_val, dof, expected = chi2_contingency(table_contingence)
print(f"Test Chi-2 pour buts <= 2 (bas scores)")
print(f"Chi2: {chi2:.4f} | dof: {dof} | p-value: {p_val:.6f}")
if p_val < 0.05:
    print("  ➜ Indépendance REJETÉE. Buts faibles CORRÉLÉS.")
else:
    print("  ➜ Indépendance ACCEPTÉE. Buts faibles DÉCORRÉLÉS.")

print("=" * 60)
print("Test Chi-2 pour buts >= 2 (hauts scores)")
df_hauts_scores = df[(df["FTHG"] >= 2) & (df["FTAG"] >= 2)].copy()
df_hauts_scores["FTHG"] = df_hauts_scores["FTHG"].clip(upper=5)
df_hauts_scores["FTAG"] = df_hauts_scores["FTAG"].clip(upper=5)
table_hs = pd.crosstab(df_hauts_scores["FTHG"], df_hauts_scores["FTAG"])
chi2_hs, p_hs, dof_hs, exp_hs = chi2_contingency(table_hs)
print(f"Chi2: {chi2_hs:.4f} | dof: {dof_hs} | p-value: {p_hs:.6f}")
if p_hs < 0.05:
    print("  ➜ Indépendance REJETÉE. Buts élevés CORRÉLÉS.")
else:
    print("  ➜ Indépendance ACCEPTÉE. Buts élevés DÉCORRÉLÉS.")
# =============================================================================
# CALCUL DES FORCES OFFENSIVES/DÉFENSIVES (ROLLING MAHER)
# =============================================================================
# Méthode Maher (1982) : estimation itérative pré-match des paramètres Poisson.
# Chaque équipe a :
#   - alpha_h, beta_a : forces offensives (domicile / extérieur)
#   - delta_h, gamma_a : forces défensives  (domicile / extérieur)
#
# Pour chaque match :
#   lambda_dom (buts domicile attendus)  = alpha_h * beta_a * μ_ligue_domicile
#   lambda_ext  (buts extérieur attendus) = gamma_a * delta_h * μ_ligue_extérieur
#
# Le lissage bayésien (K=5) intègre la moyenne de la ligue pour les premières données.

toutes_equipes = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

# Initialisation des accumulateurs (statistiques pré-match)
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
K = 5  # Facteur de lissage bayésien

for idx in tqdm(range(len(df)), desc="Calcul des Lambdas (Maher)"):
    h, a = home_teams[idx], away_teams[idx]

    # Moyennes de la ligue jusqu'au match actuel
    if total_matchs_ligue == 0:
        moy_ligue_dom, moy_ligue_ext = 1.5, 1.0
    else:
        moy_ligue_dom = total_buts_dom / total_matchs_ligue
        moy_ligue_ext = total_buts_ext / total_matchs_ligue
    moy_ligue_dom = max(moy_ligue_dom, 0.1)  # Sécurité contre division par 0
    moy_ligue_ext = max(moy_ligue_ext, 0.1)

    # Calcul des forces : (stats_actuelles + lissage) / (matchs + K)
    # Puis normalisation par la moyenne de ligue
    alpha_h = ((goals_marques_dom[h] + K * moy_ligue_dom) / (matches_joues_dom[h] + K)) / moy_ligue_dom
    delta_h = ((goals_encaisses_dom[h] + K * moy_ligue_ext) / (matches_joues_dom[h] + K)) / moy_ligue_ext
    beta_a = ((goals_encaisses_ext[a] + K * moy_ligue_dom) / (matches_joues_ext[a] + K)) / moy_ligue_dom
    gamma_a = ((goals_marques_ext[a] + K * moy_ligue_ext) / (matches_joues_ext[a] + K)) / moy_ligue_ext

    # Prédiction Poisson
    lambda_dom_list.append(alpha_h * beta_a * moy_ligue_dom)
    lambda_ext_list.append(gamma_a * delta_h * moy_ligue_ext)

    # Mise à jour des statistiques après le match
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


# =============================================================================
# MODÈLE DIXON-COLES : CORRECTION DES PETITS SCORES
# =============================================================================
# Dixon & Coles (1997) corrigent la dépendance Poisson sur les petits scores.
# Facteur correctif τ (tau) pour les événements (0,0), (0,1), (1,0), (1,1).

def log_L_neg_dixon(rho_param, mu_array, nu_array, y_home, y_away):
    """Vraisemblance négative du modèle Dixon-Coles stable.

    Paramètres
    ----------
    rho_param : array, shape (1,)
        Paramètre de corrélation Dixon-Coles à estimer.
    mu_array : array
        Paramètres Poisson λ pour domicile (E[FTHG]).
    nu_array : array
        Paramètres Poisson λ pour extérieur (E[FTAG]).
    y_home : array
        Buts observés domicile (FTHG).
    y_away : array
        Buts observés extérieur (FTAG).

    Retours
    -------
    float
        Vraisemblance négative. Pénalité (1e10) si tau ≤ 0 (infaisable).
    """
    rho_val = rho_param[0]
    total_nll = 0.0
    for k in range(len(y_home)):
        x, y = int(y_home[k]), int(y_away[k])
        mu, nu = mu_array[k], nu_array[k]
        # Correction Dixon-Coles sur les petits scores seulement
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
            return 1e10  # Infaisable : rho hors bornes
        total_nll -= np.log(tau_val)
    return total_nll


# Optimisation de rho (paramètre unique)
# On calcule d'abord les bornes valides de rho pour chaque match
borne_inf = np.maximum(-1 / df["mu_dom"].values, -1 / df["nu_ext"].values)
borne_sup = np.minimum(1 / (df["mu_dom"].values * df["nu_ext"].values), 1.0)
solution = minimize(
    fun=log_L_neg_dixon,
    x0=[0.0],
    args=(df["mu_dom"].values, df["nu_ext"].values, df["FTHG"].values, df["FTAG"].values),
    method='L-BFGS-B',
    bounds=[(np.max(borne_inf), np.min(borne_sup))]
)
rho_optimal = solution.x[0]
print(f"\n✓ Rho optimal Dixon-Coles : {rho_optimal:.4f} (Succès : {solution.success})")

# =============================================================================
# INFÉRENCE : PROBABILITÉS 1X2 (MAHER vs DIXON-COLES)
# =============================================================================
# Pour chaque match, on calcule P(Home win | λ_dom, λ_ext)
# En construisant la matrice jointe des buts possibles.

max_g = 11
buts_possibles = np.arange(max_g)
mu_arr = df["mu_dom"].values
nu_arr = df["nu_ext"].values

p1_m, pX_m, p2_m = [], [], []
p1_dc, pX_dc, p2_dc = [], [], []


def tau_correction(x, y, mu, nu, rho):
    """Facteur correctif Dixon-Coles pour les petits scores.

    Paramètres
    ----------
    x : int
        Buts domicile.
    y : int
        Buts extérieur.
    mu : float
        Paramètre Poisson domicile.
    nu : float
        Paramètre Poisson extérieur.
    rho : float
        Corrélation Dixon-Coles.

    Retours
    -------
    float
        Facteur multiplicatif τ (tau) pour la probabilité jointe.
        τ = 1 si (x,y) ∉ {(0,0), (0,1), (1,0), (1,1)}.
    """
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    if x == 0 and y == 1:
        return 1 + mu * rho
    if x == 1 and y == 0:
        return 1 + nu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


# Boucle sur les matchs pour construire les matrices jointes
for idx in range(len(df)):
    mu_val, nu_val = mu_arr[idx], nu_arr[idx]
    # Grille PMF Poisson pour chaque combinaison de buts
    p_dom = poisson.pmf(buts_possibles, mu_val)
    p_ext = poisson.pmf(buts_possibles, nu_val)

    # --- MAHER : produit Poisson indépendant ---
    matrix_maher = np.outer(p_dom, p_ext)  # P(FTHG, FTAG) ~ Poisson(mu) * Poisson(nu)
    # Somme triangulaire inférieure = P(FTHG > FTAG) = P(Home win)
    p1_m.append(np.sum(np.tril(matrix_maher, -1)))
    # Diagonale = P(FTHG == FTAG) = P(Draw)
    pX_m.append(np.sum(np.diag(matrix_maher)))
    # Triangulaire supérieure = P(FTHG < FTAG) = P(Away win)
    p2_m.append(np.sum(np.triu(matrix_maher, 1)))

    # --- DIXON-COLES : produit Poisson avec correction sur (0,0), (0,1), (1,0), (1,1) ---
    matrix_dixon = np.copy(matrix_maher)
    for x in [0, 1]:
        for y in [0, 1]:
            # Application du facteur correctif
            matrix_dixon[x, y] *= tau_correction(x, y, mu_val, nu_val, rho_optimal)
    # Renormalisation (la correction ne préserve pas la somme = 1)
    matrix_dixon /= matrix_dixon.sum()

    p1_dc.append(np.sum(np.tril(matrix_dixon, -1)))
    pX_dc.append(np.sum(np.diag(matrix_dixon)))
    p2_dc.append(np.sum(np.triu(matrix_dixon, 1)))

df["p_1_Maher"], df["p_X_Maher"], df["p_2_Maher"] = p1_m, pX_m, p2_m
df["p_1_DC"], df["p_X_DC"], df["p_2_DC"] = p1_dc, pX_dc, p2_dc

# =============================================================================
# FILTRE "WARM-UP" : EXCLUSION DES PREMIERS MATCHS
# =============================================================================
# Les estimations Rolling Maher sont instables pour les premières rencontres.
# On exclut les 4 premiers matchs domicile et extérieur de chaque équipe.

match_nb_home = df.groupby("HomeTeam").cumcount()
match_nb_away = df.groupby("AwayTeam").cumcount()
df_fiable = df[(match_nb_home >= 4) & (match_nb_away >= 4)].reset_index(drop=True)
print(f"✓ Matchs conservés après warm-up : {len(df_fiable)} / {len(df)} matchs")

# =============================================================================
# ÉVALUATION MAHER vs DIXON-COLES (MÉTRIQUES ML)
# =============================================================================

# Conversion des résultats en labels 1X2
# 0 = Home win, 1 = Draw, 2 = Away win
y_true_1X2 = np.where(df_fiable["FTHG"] > df_fiable["FTAG"], 0,
                      np.where(df_fiable["FTHG"] == df_fiable["FTAG"], 1, 2))
# Conversion binaire : 1 = Home win, 0 sinon
y_true_dom = (y_true_1X2 == 0).astype(int)

# Récupération et normalisation des probabilités
probs_maher = df_fiable[["p_1_Maher", "p_X_Maher", "p_2_Maher"]].values
probs_maher = probs_maher / probs_maher.sum(axis=1, keepdims=True)
probs_dixon = df_fiable[["p_1_DC", "p_X_DC", "p_2_DC"]].values
probs_dixon = probs_dixon / probs_dixon.sum(axis=1, keepdims=True)

# Métriques de probabilité calibrées
print("\n" + "=" * 60)
print("MÉTRIQUES PROBABILISTES (1X2)")
print("=" * 60)
print(f"Log-Loss   | Maher: {log_loss(y_true_1X2, probs_maher):.4f} "
      f"| Dixon: {log_loss(y_true_1X2, probs_dixon):.4f}")
print(f"Brier (H)  | Maher: {brier_score_loss(y_true_dom, probs_maher[:, 0]):.4f} "
      f"| Dixon: {brier_score_loss(y_true_dom, probs_dixon[:, 0]):.4f}")

# =============================================================================
# ÉVALUATION DES PRÉDICTIONS DE BUTS
# =============================================================================
# Qualité des estimations λ_dom et λ_ext via des métriques de comptage.

buts_reels_dom = df_fiable["FTHG"].values
buts_reels_ext = df_fiable["FTAG"].values
mu_attendus = df_fiable["mu_dom"].values
nu_attendus = df_fiable["nu_ext"].values

print("\n" + "=" * 60)
print("PRÉDICTION DE BUTS (DOMICILE vs EXTÉRIEUR)")
print("=" * 60)
print("DOMICILE :")
print(f"  Poisson Dev: {mean_poisson_deviance(buts_reels_dom, mu_attendus):.4f} "
      f"| MSE: {mean_squared_error(buts_reels_dom, mu_attendus):.4f} "
      f"| MAE: {mean_absolute_error(buts_reels_dom, mu_attendus):.4f}")
print("EXTÉRIEUR :")
print(f"  Poisson Dev: {mean_poisson_deviance(buts_reels_ext, nu_attendus):.4f} "
      f"| MSE: {mean_squared_error(buts_reels_ext, nu_attendus):.4f} "
      f"| MAE: {mean_absolute_error(buts_reels_ext, nu_attendus):.4f}")

# =============================================================================
# BACKTEST VALUE BETTING
# =============================================================================
# Stratégie : miser (fixed 10€) si P(modèle) > P(bookmaker implied).
# Évaluation financière : Profit Net, ROI (%).

def simuler_performance_financiere(y_true, probs_modele, cotes, nom_modele, mise=10.0):
    """Simule la performance financière d'une stratégie value betting.

    Paramètres
    ----------
    y_true : array, binary
        Résultats observés (1 = succès, 0 = échec).
    probs_modele : array
        Probabilités prédites par le modèle.
    cotes : array
        Cotes du bookmaker.
    nom_modele : str
        Nom pour l'affichage.
    mise : float, default 10.0
        Montant par pari.

    Retours
    -------
    tuple
        (nom, nb_paris, total_investi, profit_net, roi_pourcent)
    """
    implied_prob = 1 / cotes  # Probabilité implicite du bookmaker
    # Décision : parier si P(modèle) > P(bookmaker)
    preds_decision = (probs_modele > implied_prob).astype(int)
    mask = (preds_decision == 1)
    total_investi = mask.sum() * mise

    if total_investi == 0:
        return nom_modele, 0, 0.0, 0.0, 0.0

    # Gains = mise * cote si résultat correct, sinon 0
    gains = np.where(y_true[mask] == 1, mise * cotes[mask], 0.0)
    profit_net = gains.sum() - total_investi
    roi = (profit_net / total_investi) * 100

    return nom_modele, mask.sum(), total_investi, profit_net, roi


cotes_home = df_fiable.get("B365H", df_fiable.get("AvgH", 2.0)).values
res_M = simuler_performance_financiere(y_true_dom, df_fiable["p_1_Maher"].values, cotes_home, "MAHER")
res_D = simuler_performance_financiere(y_true_dom, df_fiable["p_1_DC"].values, cotes_home, "DIXON-COLES")

print("\n" + "=" * 60)
print("BACKTEST VALUE BETTING (mise fixe 10€)")
print("=" * 60)
print(f"[{res_M[0]}]       "
      f"nb paris: {res_M[1]} | Investi: {res_M[2]:.0f}€ | "
      f"Profit: {res_M[3]:.2f}€ | ROI: {res_M[4]:.2f}%")
print(f"[{res_D[0]}] "
      f"nb paris: {res_D[1]} | Investi: {res_D[2]:.0f}€ | "
      f"Profit: {res_D[3]:.2f}€ | ROI: {res_D[4]:.2f}%")

# =============================================================================
# DIXON-COLES TEMPOREL : AMÉLIORATION VIA PONDÉRATION DÉCROISSANTE
# =============================================================================
# Variante temporelle : matches récents plus importants que les anciens.
# Pondération : φ(t) = exp(-t * ξ), où t = jours depuis le dernier match / 3.5
# Optimisation conjointe de ρ (corrélation) et ξ (decay rate).

date_max_fiable = df_fiable["Date"].max()
# t en "demi-semaines" (plus fin que les semaines pour le foot) cf document Dixon
df_fiable["t_semi_weeks"] = (date_max_fiable - df_fiable["Date"]).dt.days / 3.5


def log_L_neg_dixon_global_vectorized_fixed(params, mu_array, nu_array, y_home, y_away, t_array):
    """Vraisemblance négative vectorisée du modèle Dixon-Coles temporel.

    Paramètres
    ----------
    params : array, shape (3,)
        [rho, xi, coef_dom] : corrélation, decay rate, coefficient domicile.
    mu_array : array
        Paramètres Poisson domicile (E[FTHG]).
    nu_array : array
        Paramètres Poisson extérieur (E[FTAG]).
    y_home : array
        Buts observés domicile.
    y_away : array
        Buts observés extérieur.
    t_array : array
        Temps écoulé (demi-semaines) depuis le dernier match.

    Retours
    -------
    float
        Vraisemblance négative pondérée + pénalités pour infaisabilité.

    Notes
    -----
    - Pondération : φ(t) = exp(-t * ξ)
    - Pénalités si ξ < 0 ou τ ≤ 0
    - Normalisation : coef_dom ≈ 1.2 capture l'avantage domicile additionnel
    """
    rho_val, xi_val, coef_dom = params[0], params[1], params[2]
    penalite = 0.0

    # Ajustement dynamique de l'avantage domicile
    mu_array = mu_array * coef_dom

    # Sécurité : xi doit rester positif (décroissance)
    if xi_val < 0:
        penalite += 1e7 * abs(xi_val)
        xi_val = 0.0

    # Calcul vectorisé des corrections Dixon-Coles
    tau_val = np.ones_like(y_home, dtype=float)
    m_0_0 = (y_home == 0) & (y_away == 0)
    m_0_1 = (y_home == 0) & (y_away == 1)
    m_1_0 = (y_home == 1) & (y_away == 0)
    m_1_1 = (y_home == 1) & (y_away == 1)

    tau_val[m_0_0] = 1 - mu_array[m_0_0] * nu_array[m_0_0] * rho_val
    tau_val[m_0_1] = 1 + mu_array[m_0_1] * rho_val
    tau_val[m_1_0] = 1 + nu_array[m_1_0] * rho_val
    tau_val[m_1_1] = 1 - rho_val

    # Vérification faisabilité
    if np.any(tau_val <= 0):
        return 1e8 + np.sum(np.abs(tau_val[tau_val <= 0])) + penalite

    # Vraisemblance par match
    log_match = (np.log(tau_val)
                 + poisson.logpmf(y_home, mu_array)
                 + poisson.logpmf(y_away, nu_array))

    if not np.all(np.isfinite(log_match)):
        return 1e8 + penalite

    # Pondération temporelle : matches récents plus importants
    phi = np.exp(-t_array * xi_val)
    return -np.sum(phi * log_match) + penalite


# Préparation des données filtrées
X_mu_fiable = df_fiable["mu_dom"].values
X_nu_fiable = df_fiable["nu_ext"].values
y_h_fiable = df_fiable["FTHG"].values
y_a_fiable = df_fiable["FTAG"].values
t_fiable = df_fiable["t_semi_weeks"].values

# Optimisation conjointe
solution_globale = minimize(
    fun=log_L_neg_dixon_global_vectorized_fixed,
    x0=[0.0, 0.0065, 1.2],
    args=(X_mu_fiable, X_nu_fiable, y_h_fiable, y_a_fiable, t_fiable),
    method='L-BFGS-B',
    bounds=[(-0.25, 0.10), (0.0, 0.05), (1.0, 2.0)]
)

rho_optimal, xi_optimal, coef_dom_opit = solution_globale.x
print(f"\n Dixon-Coles Temporel optimisé :")
print(f"    Rho : {rho_optimal:.4f} | Xi : {xi_optimal:.6f} | Coef_DOM : {coef_dom_opit:.4f}")

# =============================================================================
# INFÉRENCE DIXON-COLES TEMPOREL
# =============================================================================
# Application du modèle temporel optimisé pour générer les prédictions.
# Les paramètres ρ, ξ, coef_dom sont maintenant calibrés sur les données.

p1_dct, pX_dct, p2_dct = [], [], []

for idx in range(len(df_fiable)):
    mu_val, nu_val = X_mu_fiable[idx], X_nu_fiable[idx]
    p_dom = poisson.pmf(buts_possibles, mu_val)
    p_ext = poisson.pmf(buts_possibles, nu_val)

    # Construction de la matrice jointe avec correction Dixon-Coles
    matrix_dixon_temp = np.outer(p_dom, p_ext)

    for x in [0, 1]:
        for y in [0, 1]:
            matrix_dixon_temp[x, y] *= tau_correction(x, y, mu_val, nu_val, rho_optimal)
    # Renormalisation
    matrix_dixon_temp /= np.sum(matrix_dixon_temp)

    p1_dct.append(np.sum(np.tril(matrix_dixon_temp, -1)))
    pX_dct.append(np.sum(np.diag(matrix_dixon_temp)))
    p2_dct.append(np.sum(np.triu(matrix_dixon_temp, 1)))

df_fiable["p_1_DC_Temp"] = p1_dct
df_fiable["p_X_DC_Temp"] = pX_dct
df_fiable["p_2_DC_Temp"] = p2_dct

probs_dc_temp = df_fiable[["p_1_DC_Temp", "p_X_DC_Temp", "p_2_DC_Temp"]].values

# =============================================================================
# COMPARAISON FINALE : MAHER vs DIXON-COLES vs DIXON-COLES TEMPOREL
# =============================================================================

print("\n" + "=" * 60)
print("COMPARAISON FINALE (MAHER vs DIXON-COLES vs DIXON-COLES TEMPOREL)")
print("=" * 60)
print(f"Log-Loss   | Maher: {log_loss(y_true_1X2, probs_maher):.4f} "
      f"| Dixon: {log_loss(y_true_1X2, probs_dixon):.4f} "
      f"| Dixon Temp: {log_loss(y_true_1X2, probs_dc_temp):.4f}")
print(f"Brier (H)  | Maher: {brier_score_loss(y_true_dom, probs_maher[:, 0]):.4f} "
      f"| Dixon: {brier_score_loss(y_true_dom, probs_dixon[:, 0]):.4f} "
      f"| Dixon Temp: {brier_score_loss(y_true_dom, probs_dc_temp[:, 0]):.4f}")

# Backtest sur Dixon-Coles Temporel
res_DCT = simuler_performance_financiere(y_true_dom, df_fiable["p_1_DC_Temp"].values,
                                          cotes_home, "DIXON-COLES TEMP")
print("\n" + "=" * 60)
print("BACKTEST FINAL (mise fixe 10€)")
print("=" * 60)
print(f"[{res_M[0]}]            "
      f"nb paris: {res_M[1]} | Investi: {res_M[2]:.0f}€ | "
      f"Profit: {res_M[3]:.2f}€ | ROI: {res_M[4]:.2f}%")
print(f"[{res_D[0]}]      "
      f"nb paris: {res_D[1]} | Investi: {res_D[2]:.0f}€ | "
      f"Profit: {res_D[3]:.2f}€ | ROI: {res_D[4]:.2f}%")
print(f"[{res_DCT[0]}] "
      f"nb paris: {res_DCT[1]} | Investi: {res_DCT[2]:.0f}€ | "
      f"Profit: {res_DCT[3]:.2f}€ | ROI: {res_DCT[4]:.2f}%")

# =============================================================================
# VISUALISATION : LOG-LOSS CUMULÉE EN SÉRIE TEMPORELLE
# =============================================================================
# Pour chaque modèle, on calcule la log-loss par match et on moyenne cumulativement.
# Chaque courbe montre la performance aggradative du modèle au fil du temps.
# Pente descendante = amélioration (vers des meilleurs log-loss).

# Calcul des pertes de log pour chaque match
losses_maher = [-np.log(probs_maher[i, y]) for i, y in enumerate(y_true_1X2)]
losses_dixon = [-np.log(probs_dixon[i, y]) for i, y in enumerate(y_true_1X2)]
losses_dc_temp = [-np.log(probs_dc_temp[i, y]) for i, y in enumerate(y_true_1X2)]

# Moyens cumulatifs : lissage et convergence progressifs
cum_loss_maher = np.cumsum(losses_maher) / (np.arange(len(losses_maher)) + 1)
cum_loss_dixon = np.cumsum(losses_dixon) / (np.arange(len(losses_dixon)) + 1)
cum_loss_dc_temp = np.cumsum(losses_dc_temp) / (np.arange(len(losses_dc_temp)) + 1)

# Abscisse : dates (ou index si pas de colonne Date)
axes_x = df_fiable["Date"] if "Date" in df_fiable.columns else np.arange(len(df_fiable))

# Configuration des séries à tracer : (données, couleur, linestyle, linewidth, label)
series = [
    (cum_loss_maher, "#c0392b", "-", 1.5, f"Maher ({cum_loss_maher[-1]:.4f})"),
    (cum_loss_dixon, "#2980b9", "-", 1.5, f"Dixon-Coles ({cum_loss_dixon[-1]:.4f})"),
    (cum_loss_dc_temp, "#27ae60", "-", 2.2, f"Dixon-Coles Temporel ({cum_loss_dc_temp[-1]:.4f})"),
]

# Boucle de visualisation : une figure par série
for (data, color, ls, lw, label) in series:
    plt.figure(figsize=(10, 4))

    plt.plot(axes_x, data, color=color, lw=lw, ls=ls, label=label)

    # Annotation de la valeur finale à droite
    plt.annotate(f"{data[-1]:.4f}",
                 xy=(axes_x.iloc[-1], data[-1]),
                 xycoords="data", fontsize=9, color=color, fontweight="bold",
                 va="center", ha="left", xytext=(8, 0), textcoords="offset points")

    plt.title(f"Évolution du Log-Loss cumulé moyen — {label.split(' (')[0]}",
              fontsize=11, fontweight="bold")
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Log-Loss Moyen", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9)

    # Ajustement automatique des marges verticales pour éviter les chevauchements
    margin = (max(data) - min(data)) * 0.1 if len(data) > 1 else 0.1
    plt.ylim(min(data) - margin, max(data) + margin)

    plt.tight_layout()
    plt.show()
