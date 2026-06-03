# %%
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, f1_score, mean_squared_error, confusion_matrix
import seaborn as sns

sys.path.append(os.path.abspath('code'))
import config as conf

DATA_PATH = conf.path_clean_cell
MODELS_DIR = "models"
SPLIT_RATIO = 0.8
TARGET_BIN = "Hvs"

MODELS_CONFIG = {
    "LogisticRegression (Base)": ("LogisticRegression_Base", True, 0.68),
    "LogisticRegression (Optimisé)": ("LogisticRegression_Optimisé", True, 0.68),
    "RandomForest": ("RandomForest", False, 0.81),
    "RandomForest (optimisé)": ("RandomForest_optimisé", False, 0.67),
    "XGBoost (Base)": ("XGBoost_Base", False, 0.75),
    "XGBoost (Optimisé)": ("XGBoost_Optimisé", False, 0.66),
    "SVM RBF": ("SVM_RBF", True, 0.68),
    "LightGBM": ("LightGBM", False, 0.73),
    "LightGBM (Optimisé)": ("LightGBM_Optimisé", False, 0.68),
}

# %% =========================================================================
# CHARGEMENT DES DONNÉES ET ALIGNEMENT DU TEST CHRONOLOGIQUE
# =========================================================================

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

split_idx = int(len(df) * SPLIT_RATIO)

# Données brutes de test
df_test = df.iloc[split_idx:].copy()
X_test_brut = df_test[conf.ft_commune]
y_test = df_test[TARGET_BIN].values
cotes_test = df_test.get("BWH", df_test.get("AvgH", 2.0)).values


# %% =========================================================================
# chargement des modèles
# =========================================================================
def _load_models(models_dir=MODELS_DIR):
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler introuvable à l'emplacement : {scaler_path}")

    scaler = joblib.load(scaler_path)
    print(f"\nChargement des modèles ")

    trained = {}
    for nom_affiche, (nom_fichier, use_scaling, seuil_opt) in MODELS_CONFIG.items():
        path = os.path.join(models_dir, f"{nom_fichier}.pkl")
        if os.path.exists(path):
            trained[nom_affiche] = (joblib.load(path), scaler, use_scaling, seuil_opt)
            print(f"    [OK]  {nom_affiche}")
        else:
            print(f"    [--]  {nom_affiche} (fichier '{path}' introuvable, ignoré)")

    if not trained:
        raise FileNotFoundError(f"Aucun modèle chargé depuis '{models_dir}/'. Checkez vos noms de fichiers.")
    return trained


# Exécution du chargement
modeles_ml = _load_models(MODELS_DIR)

# %% =========================================================================
#  MODÈLE STATISTIQUE (DIXON-COLES) SUR LE TEST
# =========================================================================
toutes_equipes = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

# Compteurs pour le calcul dynamique (Rolling Maher)
goals_m_dom = {t: 0 for t in toutes_equipes}
goals_e_dom = {t: 0 for t in toutes_equipes}
goals_m_ext = {t: 0 for t in toutes_equipes}
goals_e_ext = {t: 0 for t in toutes_equipes}
m_joues_dom = {t: 0 for t in toutes_equipes}
m_joues_ext = {t: 0 for t in toutes_equipes}
tot_b_dom, tot_b_ext, tot_m = 0, 0, 0
K = 5

mu_tous, nu_tous = [], []
home_teams, away_teams = df["HomeTeam"].values, df["AwayTeam"].values
fthg, ftag = df["FTHG"].values, df["FTAG"].values

for idx in range(len(df)):
    h, a = home_teams[idx], away_teams[idx]

    moy_dom = tot_b_dom / tot_m if tot_m > 0 else 1.5
    moy_ext = tot_b_ext / tot_m if tot_m > 0 else 1.0

    alpha_h = ((goals_m_dom[h] + K * moy_dom) / (m_joues_dom[h] + K)) / moy_dom
    delta_h = ((goals_e_dom[h] + K * moy_ext) / (m_joues_dom[h] + K)) / moy_ext
    beta_a = ((goals_e_ext[a] + K * moy_dom) / (m_joues_ext[a] + K)) / moy_dom
    gamma_a = ((goals_m_ext[a] + K * moy_ext) / (m_joues_ext[a] + K)) / moy_ext

    mu_tous.append(alpha_h * beta_a * moy_dom)
    nu_tous.append(gamma_a * delta_h * moy_ext)

    goals_m_dom[h] += fthg[idx]
    goals_e_dom[h] += ftag[idx]
    goals_m_ext[a] += ftag[idx]
    goals_e_ext[a] += fthg[idx]
    m_joues_dom[h] += 1
    m_joues_ext[a] += 1
    tot_b_dom += fthg[idx]
    tot_b_ext += ftag[idx]
    tot_m += 1

# Extraction des lambdas pour le bloc Train afin d'optimiser rho sans Data Leakage
mu_train = np.array(mu_tous[:split_idx])
nu_train = np.array(nu_tous[:split_idx])
fthg_train = fthg[:split_idx]
ftag_train = ftag[:split_idx]


def tau(x, y, mu, nu, rho):
    if x == 0 and y == 0: return 1 - mu * nu * rho
    if x == 0 and y == 1: return 1 + mu * rho
    if x == 1 and y == 0: return 1 + nu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def log_L_neg(params, mu, nu, x, y):
    rho = params[0]
    if abs(rho) > 1: return 1e6

    t_factors = np.ones_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] <= 1 and y[i] <= 1:
            t_factors[i] = tau(x[i], y[i], mu[i], nu[i], rho)

    if np.any(t_factors <= 0): return 1e6

    log_lik = np.log(poisson.pmf(x, mu)) + np.log(poisson.pmf(y, nu)) + np.log(t_factors)
    return -np.sum(log_lik)


# Optimisation de rho uniquement sur le Train pour respecter la chronologie
print("Optimisation du paramètre de dépendance rho sur le Train")
res = minimize(fun=log_L_neg, x0=[0.0], args=(mu_train, nu_train, fthg_train, ftag_train), method='Nelder-Mead')
rho_optimal = res.x[0]
print(f"    [OK] rho optimal calculé : {rho_optimal:.4f}")

# Application sur le jeu de Test
mu_test = mu_tous[split_idx:]
nu_test = nu_tous[split_idx:]

probs_dc_home = []
max_g = 10
buts = np.arange(max_g)

for i in range(len(df_test)):
    mu_val, nu_val = mu_test[i], nu_test[i]
    matrix = np.outer(poisson.pmf(buts, mu_val), poisson.pmf(buts, nu_val))
    for x in [0, 1]:
        for y in [0, 1]:
            matrix[x, y] *= tau(x, y, mu_val, nu_val, rho_optimal)
    probs_dc_home.append(np.sum(np.tril(matrix, -1)))

probs_dc_home = np.array(probs_dc_home)

# %% =========================================================================
# SYNTHÈSE DES COMPARAISONS
# =========================================================================
dict_profits_cumules = {}
dict_nb_paris = {}  # Stockage propre et exact du nombre total de paris
dict_y_pred_binaires = {}  # Contiendra les vecteurs y_pred pour les matrices de confusion


def simuler_value_bet(probs_pred, cotes, y_true, nom_modele, critere_decision, est_ml=True, mise=10):
    prob_bookmaker = 1 / cotes

    if est_ml:
        # Logique de masque y_p == 1 par rapport au seuil_opt
        mask = (probs_pred >= critere_decision)
    else:
        # Pour Dixon-Coles : y_p == 1 si supérieur à la proba bookmaker
        mask = (probs_pred > prob_bookmaker)

    total_paris = mask.sum()
    total_investi = total_paris * mise

    gains_par_match = np.where(mask & (y_true == 1), mise * cotes, 0.0)
    mises_par_match = np.where(mask, mise, 0.0)
    profits_par_match = gains_par_match - mises_par_match

    dict_profits_cumules[nom_modele] = np.cumsum(profits_par_match)
    dict_y_pred_binaires[nom_modele] = mask.astype(int)
    dict_nb_paris[nom_modele] = total_paris

    if total_investi == 0:
        return {"Modèle": nom_modele, "Total Mise ": 0, "Profit Net ": 0.0, "ROI (%)": 0.0}

    profit_net = profits_par_match.sum()

    return {
        "Modèle": nom_modele,
        "Total Mise ": int(total_investi),
        "Profit Net ": round(profit_net, 2),
        "ROI (%)": round((profit_net / total_investi) * 100, 2)
    }


# Simulation Dixon-Coles (Pour le F1 score, y_pred_bin utilise le critère implicite proba > bookmaker)
y_pred_bin_dc = (probs_dc_home > (1 / cotes_test)).astype(int)

bilan_ds = [{
    "Modèle": "Dixon-Coles (Stats)",
    "Log-Loss": round(log_loss(y_test, probs_dc_home), 4),
    "MSE": round(mean_squared_error(y_test, probs_dc_home), 4),
    "F1 Score": round(f1_score(y_test, y_pred_bin_dc, zero_division=0), 4),
    "AUC ROC": round(roc_auc_score(y_test, probs_dc_home), 4)
}]

bilan_finances = [
    simuler_value_bet(probs_dc_home, cotes_test, y_test, "Dixon-Coles (Stats)", critere_decision=None, est_ml=False)]

# Évaluation des modèles de Machine Learning chargés
for nom_affiche, (model, scaler, use_scaling, seuil_opt) in modeles_ml.items():
    X_input = scaler.transform(X_test_brut) if use_scaling else X_test_brut.values
    probs_ml = model.predict_proba(X_input)[:, 1]

    # y_pred binaire basé strictement sur le seuil_opt du modèle
    y_pred_bin_ml = (probs_ml >= seuil_opt).astype(int)

    bilan_ds.append({
        "Modèle": nom_affiche,
        "Log-Loss": round(log_loss(y_test, probs_ml), 4),
        "MSE": round(mean_squared_error(y_test, probs_ml), 4),
        "F1 Score": round(f1_score(y_test, y_pred_bin_ml, zero_division=0), 4),
        "AUC ROC": round(roc_auc_score(y_test, probs_ml), 4)
    })

    bilan_finances.append(
        simuler_value_bet(probs_ml, cotes_test, y_test, nom_affiche, critere_decision=seuil_opt, est_ml=True))

# %% =========================================================================
# AFFICHAGE DES TABLEAUX DE RÉSULTATS
# =========================================================================
df_ds = pd.DataFrame(bilan_ds)
print("\n" + "=" * 90)
print("     COMPARAISON DES PERFORMANCES ")
print("=" * 90)
print(df_ds.sort_values('MSE', ascending=True).to_string(index=False))

print("\n" + "=" * 90)
print("     COMPARAISON DES PERFORMANCES FINANCIÈRES ")
print("=" * 90)
df_finances_res = pd.DataFrame(bilan_finances).sort_values('ROI (%)', ascending=False)
print(df_finances_res.to_string(index=False))

# %% =========================================================================
#
# --- GRAPHIQUE 1 : ÉVOLUTION CHRONOLOGIQUE DES PROFITS ---
plt.figure(figsize=(14, 6))
index_matchs = np.arange(1, len(df_test) + 1)

# Courbe Dixon-Coles
p_final_dc = dict_profits_cumules["Dixon-Coles (Stats)"][-1]
n_paris_dc = dict_nb_paris["Dixon-Coles (Stats)"]
plt.plot(index_matchs, dict_profits_cumules["Dixon-Coles (Stats)"],
         label=f"Dixon-Coles (Stats) | Profit: {p_final_dc:.1f}€ ({n_paris_dc} paris)",
         linewidth=2, color="blue")

# Courbes ML
for nom_modele in dict_profits_cumules.keys():
    if nom_modele == "Dixon-Coles (Stats)": continue
    profit_serie = dict_profits_cumules[nom_modele]
    profit_final = profit_serie[-1]
    nb_paris = dict_nb_paris[nom_modele]
    if profit_serie[0] != 0: nb_paris += 1

    plt.plot(index_matchs, profit_serie, label=f"{nom_modele} | Profit: {profit_final:.1f}€ ({nb_paris} paris)",
             alpha=0.75, linestyle="--")

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.title("Évolution du Profit Cumulé au fil des Matchs (Jeu de Test)", fontsize=13, fontweight='bold')
plt.xlabel("Nombre de Matchs Évalués", fontsize=11)
plt.ylabel("Profit Net Cumulé (€)", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
plt.tight_layout()
plt.show()

# %% =========================================================================
metrics_to_plot = ["MSE", "F1 Score", "AUC ROC"]

# Création d'une grille de 1 ligne et 3 colonnes
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=False)

# Définition de couleurs distinctes pour chaque métrique pour le rendu visuel
colors = ["#ff7f0e", "#2ca02c", "#d62728"]

for i, metric in enumerate(metrics_to_plot):
    ax = axs[i]

    # Extraction des données triées dans le même ordre que le DataFrame d'origine
    modeles = df_ds["Modèle"]
    valeurs = df_ds[metric]

    # Tracé des barres pour le subplot courant
    bars = ax.bar(modeles, valeurs, color=colors[i], edgecolor='k', alpha=0.85)

    # Ajout des étiquettes de valeurs au-dessus de chaque barre
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Décalage de 3 points vers le haut
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Personnalisation de l'axe courant
    ax.set_title(f"Comparatif : {metric}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score / Valeur", fontsize=10)
    ax.set_xticklabels(modeles, rotation=30, ha="right", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Ajustement des limites pour laisser de la place aux annotations textuelles
    ax.set_ylim(0, max(valeurs) * 1.15)

plt.tight_layout()
plt.show()

# %% =========================================================================
#  MATRICES DE CONFUSION
# =========================================================================
# On récupère tous les modèles qui ont fait des prédictions binaires
liste_modeles = list(dict_y_pred_binaires.keys())
num_modeles = len(liste_modeles)

nb_cols = 4
nb_lignes = int(np.ceil(num_modeles / nb_cols))

fig, axes = plt.subplots(nb_lignes, nb_cols, figsize=(20, 4 * nb_lignes))
axes = axes.flatten()

for idx, nom_modele in enumerate(liste_modeles):
    ax = axes[idx]
    y_pred_bin = dict_y_pred_binaires[nom_modele]

    # 1. Génération de la matrice brute
    cm = confusion_matrix(y_test, y_pred_bin)

    # 2. Normalisation par ligne (Recall/Sensibilité par classe)
    # Le epsilon (1e-9) évite la division par zéro si une ligne est complètement vide
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # 3. Affichage de la Heatmap avec le format pourcentage
    sns.heatmap(cm_percent, annot=True, fmt='.1%', cmap='Blues', cbar=False, ax=ax,
                square=True, annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['Non-Victoire', 'Victoire Dom'],
                yticklabels=['Non-Victoire', 'Victoire Dom'],
                vmin=0, vmax=1)  # Fixe l'échelle des couleurs entre 0 et 100%

    ax.set_title(f"{nom_modele}\n(Paris émis : {dict_nb_paris[nom_modele]})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Classe Prédite", fontsize=9)
    ax.set_ylabel("Classe Réelle", fontsize=9)

for idx in range(num_modeles, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle("Matrices de Confusion Test ",
             fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()

# %% =========================================================================
# GRAPH COMPLÉMENTAIRE : ÉVOLUTION CHRONOLOGIQUE DU LOG-LOSS CUMULÉ
# =========================================================================

# 1. Dictionnaire pour stocker les probabilités de tous les modèles sur le Test
dict_probs_tous = {
    "Dixon-Coles (Stats)": probs_dc_home
}

# Extraction des probabilités pour les modèles ML
for nom_affiche, (model, scaler, use_scaling, _) in modeles_ml.items():
    X_input = scaler.transform(X_test_brut) if use_scaling else X_test_brut.values
    probs_ml = model.predict_proba(X_input)[:, 1]
    dict_probs_tous[nom_affiche] = probs_ml

# 2. Calcul des courbes de Log-Loss cumulé (Moyenne glissante)
plt.figure(figsize=(14, 6))
index_matchs = np.arange(1, len(df_test) + 1)

# Style pour mettre en valeur Dixon-Coles par rapport à la masse de modèles ML
styles_config = {
    "Dixon-Coles (Stats)": {"color": "blue", "linewidth": 2.5, "zorder": 10, "linestyle": "-"}
}

for nom_modele, probs_pred in dict_probs_tous.items():
    # Calcul du log-loss individuel par match (labels [0, 1] car binaire Hvs)
    losses_indiv = [log_loss([y], [p], labels=[0, 1]) for y, p in zip(y_test, probs_pred)]

    # Calcul de la moyenne cumulée : somme_cumulée / nombre_de_matchs
    cum_loss = np.cumsum(losses_indiv) / index_matchs

    # Configuration graphique personnalisée ou par défaut
    cfg = styles_config.get(nom_modele, {"alpha": 0.6, "linestyle": "--", "linewidth": 1.2})

    plt.plot(index_matchs, cum_loss, label=f"{nom_modele} (Final: {cum_loss[-1]:.4f})", **cfg)

plt.title("Stabilité et Évolution du Log-Loss Cumulé sur le Jeu de Test (Plus bas = Meilleur)",
          fontsize=13, fontweight='bold')
plt.xlabel("Nombre de Matchs Évalués (Chronologique)", fontsize=11)
plt.ylabel("Log-Loss Moyen Cumulé", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Placement de la légende à droite pour éviter de surcharger le graphique
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)
plt.tight_layout()
plt.show()

# %%
