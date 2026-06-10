"""
=============================================================================
PIPELINE DE PRÉDICTION DE MATCHS DE FOOTBALL — MODÈLES DE CLASSIFICATION
=============================================================================
Ce script entraîne et évalue plusieurs modèles de machine learning pour prédire
l'issue binaire d'un match de football (victoire à domicile ou non).

Étapes principales :
    1. Chargement et préparation des données chronologiques
    2. Définition et optimisation des modèles (via RandomizedSearchCV)
    3. Analyse financière : ROI et optimisation des seuils de confiance
    4. Visualisation des métriques de performance (ROC, PR, Calibration)

Dépendances : lightgbm, xgboost, scikit-learn, umap, seaborn, scipy
=============================================================================
"""

import os
import sys
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import friedmanchisquare, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.append(os.path.abspath('code'))
import config as conf

print("Import")

# %%

DATA_PATH = conf.path_clean_cell
SPLIT_RATIO = 0.8
SEUIL_SECURISE = 0.65
MISE_UNITE = 10

COTES_MAP = {"Hvs": "BWH", "Avs": "BWA", "Dvs": "BWD"}

FEATURES = conf.ft_commune
TARGETS = conf.targets
TARGET_BIN = "Hvs"


# %%=============================================================================
# FONCTIONS
# =============================================================================

def split_data(X, y, ratio=SPLIT_RATIO):
    """
    Découpe les données en ensembles d'entraînement et de test en respectant
    l'ordre chronologique

    Parameters
    -
    X : pd.DataFrame
        Matrice de features.
    y : pd.Series
        Vecteur cible.
    ratio : float, optional
        Proportion des données allouées à l'entraînement. Par défaut : 0.8.
    Returns
    -
    tuple : (X_train, X_test, y_train, y_test)
    """
    sp = int(len(X) * ratio)
    return X.iloc[:sp], X.iloc[sp:], y.iloc[:sp], y.iloc[sp:]


def compute_f1(y_true, y_pred, target_name):
    """
    Calcule le F1-Score en adaptant la stratégie de moyenne à la nature de la cible.

    - Cible multi-classe (ex: 'FTR' avec H/D/A) → moyenne pondérée (weighted)
    - Cible binaire (ex: 'Hvs') → moyenne binaire standard

    Parameters
    -
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.
    target_name : str
        Nom de la variable cible.

    Returns
    -
    float : Score F1.
    """
    if target_name == "FTR":
        avg = "weighted"
    else:
        avg = "binary"
    return f1_score(y_true, y_pred, average=avg, zero_division=0)


def compute_roi(y_test, y_pred, cotes, mise=MISE_UNITE):
    """
    Calcule le ROI (Retour sur Investissement) d'une stratégie de paris.

    La mise est engagée uniquement lorsque le modèle prédit une victoire (y_pred == 1).
    Le gain est calculé en multipliant la mise par la cote bookmaker si le pari est gagnant.

    Parameters
    -
    y_test : array-like
        Résultats réels des matchs (1 = victoire, 0 = autre).
    y_pred : array-like
        Prédictions du modèle (1 = pari placé, 0 = pas de pari).
    cotes : array-like
        Cotes bookmaker associées à chaque match.
    mise : float, optional
        Montant misé par pari. Par défaut : MISE_UNITE (10€).

    Returns
    -
    tuple :
        - roi (float) : ROI en pourcentage.
        - total_mise (int) : Montant total investi.
        - profit_net (float) : Profit ou perte nette en euros.
    """
    y_t = np.array(y_test)
    y_p = np.array(y_pred)
    c = np.array(cotes)

    # On ne parie que sur les matchs où le modèle prédit une victoire
    mask = (y_p == 1)
    total_mise = mask.sum() * mise

    # Cas limite : aucun pari placé
    if total_mise == 0:
        return 0.0, 0, 0.0

    gains = np.where(y_t[mask] == 1, mise * c[mask], 0.0)
    profit_net = gains.sum() - total_mise
    roi = (profit_net / total_mise) * 100

    return roi, int(total_mise), profit_net


def compute_roi_dynamique(y_test, probas, cotes):
    """
    Calcule le ROI en utilisant une stratégie de mise par paliers de confiance :
    - Probabilité entre [0.5, 0.6[  -> Mise = 10 jetons
    - Probabilité entre [0.6, 0.7[  -> Mise = 20 jetons
    - Probabilité entre [0.8, 1.0]  -> Mise = 30 jetons
    Un pari n'est validé que s'il s'agit d'un Value Bet (probas > 1/cote).
    """
    y_t = np.array(y_test)
    c = np.array(cotes)
    p = np.array(probas)
    # Condition de Value Bet standard
    implied_prob = 1 / c
    is_value = (p > implied_prob)
    # Initialisation du vecteur de mises à 0
    mises = np.zeros(len(p))
    # Application stricte de tes paliers de confiance
    mises[(p >= 0.5) & (p < 0.7) & is_value] = 10.0
    mises[(p >= 0.7) & (p <= 1.0) & is_value] = 30.0
    # Masque des matchs où on a effectivement misé
    mask = (mises > 0)
    total_mise = mises[mask].sum()
    if total_mise == 0:
        return 0.0, 0, 0.0
    # Gains calculés au prorata de la mise dynamique affectée
    gains = np.where(y_t[mask] == 1, mises[mask] * c[mask], 0.0)
    profit_net = gains.sum() - total_mise
    roi = (profit_net / total_mise) * 100

    return roi, int(total_mise), profit_net


def compute_roi_par_modele(y_test, probas, cotes, seuil_optimal):
    y_t = np.array(y_test)
    c = np.array(cotes)
    p = np.array(probas)

    prob_bookmaker = 1 / c
    is_value = (p > prob_bookmaker)
    mises = np.zeros(len(p))
    # Le premier palier se déclenche au seuil propre du modèle
    mises[(p >= seuil_optimal) & (p < seuil_optimal + 0.10) & is_value] = 10.0
    mises[(p >= seuil_optimal + 0.10) & (p <= 1.0) & is_value] = 30.0
    mask = (mises > 0)
    total_mise = mises[mask].sum()
    if total_mise == 0:
        return 0.0, 0, 0.0
    gains = np.where(y_t[mask] == 1, mises[mask] * c[mask], 0.0)
    profit_net = gains.sum() - total_mise
    roi = (profit_net / total_mise) * 100

    return roi, int(total_mise), profit_net


# %%=============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

X_brut = df[FEATURES]
X_brut_nn = df[conf.ft_nn]  # Features spécifiques au MLP

split_idx = int(len(df) * SPLIT_RATIO)

y_train_fin = df[TARGET_BIN].iloc[:split_idx]
ratio_poids = (
    (y_train_fin == 0).sum() / (y_train_fin == 1).sum()
    if (y_train_fin == 1).sum() > 0
    else 1.0
)

# %%=============================================================================
# modèles et grilles randomsearch
# =============================================================================

# Grille de recherche pour la Régression Logistique
lr_grid = {
    'penalty': ['l1', 'l2'],  # Type de régularisation
    'C': np.logspace(-3, 3, 20),  # Inverse de la force de régularisation
    'solver': ['saga'],  # Solver compatible l1/l2 sur grands datasets
    'tol': [1e-4, 1e-3, 1e-2]  # Tolérance pour le critère de convergence
}

# Grille de recherche pour XGBoost
xgb_grid = {
    'n_estimators': randint(50, 350),  # Nombre d'arbres
    'learning_rate': uniform(0.01, 0.2),  # Taux d'apprentissage
    'max_depth': [3, 4, 5, 6],  # Profondeur maximale des arbres
    'reg_alpha': uniform(0, 2),  # Régularisation L1 (Lasso)
    'reg_lambda': uniform(1, 5),  # Régularisation L2 (Ridge)
    'subsample': uniform(0.6, 0.4),  # Fraction de lignes par arbre
    'colsample_bytree': uniform(0.6, 0.4)  # Fraction de colonnes par arbre
}
lgbm_grid = {
    'n_estimators': randint(50, 350),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': [3, 4, 5, 6],
    'num_leaves': randint(20, 100),  # spécifique LightGBM
    'reg_alpha': uniform(0, 2),  # L1
    'reg_lambda': uniform(1, 5),  # L2
}

rf_grid = {
    'n_estimators': randint(50, 350),
    'max_depth': [3, 4, 5, 6],
}

# Grille de recherche pour le MLP
mlp_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50), (100, 100)],
    'alpha': uniform(1e-5, 1e-1),
    'learning_rate_init': uniform(1e-4, 1e-2),
    'batch_size': [32, 64, 128],
}

# Validation croisée temporelle (respecte l'ordre chronologique des matchs)
cv_tempo = TimeSeriesSplit(n_splits=3)

# Pipeline des modèles : chaque entrée définit le modèle, si un scaler est nécessaire,
# et si une optimisation RandomizedSearch doit être effectuée.
models_pipeline = {
    "LogisticRegression (Base)": {
        "instance": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "scaling": True,  # Nécessite une normalisation des features
        "optimize": False  # Entraînement avec paramètres par défaut
    },
    "LogisticRegression (Optimisé)": {
        "instance": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
        "scaling": True,
        "optimize": True,  # Recherche d'hyperparamètres activée
        "search_args": {
            "param_distributions": lr_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo
        }
    },
    "RandomForest": {
        "instance": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "scaling": False,  # Les forêts aléatoires ne nécessitent pas de normalisation
        "optimize": False
    },
    "RandomForest (optimisé)": {
        "instance": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": rf_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo
        }
    },
    "XGBoost (Base)": {
        "instance": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=ratio_poids,  # Compensation du déséquilibre de classes
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
        ),
        "scaling": False,
        "optimize": False
    },
    "XGBoost (Optimisé)": {
        "instance": XGBClassifier(
            scale_pos_weight=ratio_poids, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": xgb_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo
        }
    },
    "SVM RBF": {
        "instance": SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced',
            probability=True,  # Nécessaire pour predict_proba
            random_state=42
        ),
        "scaling": True,  # Les SVM sont très sensibles à l'échelle des features
        "optimize": False
    },
    "LightGBM": {
        "instance": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1
        ),
        "scaling": False,
        "optimize": False
    },
    "LightGBM (Optimisé)": {
        "instance": lgb.LGBMClassifier(
            class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1
        ),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": lgbm_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo
        }
    },
    # MLP : utilise conf.ft_nn et un scaler dédié (scaler_mlp)
    "MLP (Base)": {
        "instance": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=800, alpha=0.0001,
            solver='adam', random_state=21, tol=1e-9
        ),
        "scaling": True,
        "nn": True,  # flag : indique l'usage de ft_nn + scaler_mlp
        "optimize": False
    },
    "MLP (Optimisé)": {
        "instance": MLPClassifier(max_iter=800, solver='adam', random_state=42, tol=1e-9),
        "scaling": True,
        "nn": True,
        "optimize": True,
        "search_args": {
            "param_distributions": mlp_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo
        }
    }
}
# %%=============================================================================
# train
# =============================================================================

print("\n" + "=" * 70)
print(" train et opti")
print("=" * 70)

scaler = StandardScaler()
X_train_brut, X_test_brut, y_train, y_test = split_data(X_brut, df[TARGET_BIN])

X_train_scaled = scaler.fit_transform(X_train_brut)
X_test_scaled = scaler.transform(X_test_brut)

# Données et scaler dédiés au MLP (espace de features ft_nn)
scaler_mlp = StandardScaler()
X_train_brut_nn, X_test_brut_nn, _, _ = split_data(X_brut_nn, df[TARGET_BIN])
X_train_scaled_nn = scaler_mlp.fit_transform(X_train_brut_nn)
X_test_scaled_nn = scaler_mlp.transform(X_test_brut_nn)

# Stockage des modèles entraînés et de leurs données de test associées
trained_models = {}

for name, config in models_pipeline.items():
    is_nn = config.get("nn", False)  # True uniquement pour les MLP

    if is_nn:
        # MLP : features ft_nn + scaler_mlp
        X_tr = X_train_scaled_nn
        X_te = X_test_scaled_nn
    elif config["scaling"]:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train_brut
        X_te = X_test_brut

    if config["optimize"]:
        print(f" Optimisation RandomSearch pour : {name}")
        search = RandomizedSearchCV(
            estimator=config["instance"],
            **config["search_args"],
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_tr, y_train)
        best_model = search.best_estimator_
        print(f" Meilleurs paramètres identifiés : {search.best_params_}")
        trained_models[name] = (best_model, X_te)
    else:
        print(f" Entraînement de base pour : {name}")
        model = config["instance"].fit(X_tr, y_train)
        trained_models[name] = (model, X_te)

# %%=============================================================================
# opti seuil de confiance et analyse ROI
# =============================================================================

cotes_test_bilan = df[COTES_MAP[TARGET_BIN]].iloc[split_idx:]

seuils_a_tester = np.arange(0.40, 0.85, 0.01)

# Résultats des stratégies à seuils fixes (standard et sécurisée)
bilan_data = []
# %% baseline
# =====================================
# =====================================
preds_baseline = np.ones(len(y_test), dtype=int)
# F1
f1_baseline = compute_f1(y_test, preds_baseline, TARGET_BIN)
# ROI
probas_baseline = 1 / cotes_test_bilan  # proxy : probabilité implicite bookmaker
preds_baseline_sec = (probas_baseline >= SEUIL_SECURISE).astype(int)
roi_std_bl, mise_std_bl, profit_std_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan)
roi_sec_bl, mise_sec_bl, profit_sec_bl = compute_roi(y_test, preds_baseline_sec, cotes_test_bilan)
bilan_data.append({
    "Modèle": "Baseline",
    "ROI Std (%)": round(roi_std_bl, 2),
    "Mise Std": mise_std_bl,
    "Profit Std": round(profit_std_bl, 2),
    "ROI Sécu (%)": round(roi_sec_bl, 2),
    "Mise Sécu": mise_sec_bl,
    "Profit Sécu": round(profit_sec_bl, 2),
    "F1": round(f1_baseline, 3)
})
print(f"\n BASELINE — Accuracy : {y_test.mean():.1%} | F1 : {f1_baseline:.3f} | ROI : {roi_std_bl:.2f}")

# =====================================
# %%=====================================

# Initialisation des structures de stockage
tuning_resultats = {}

if 'f1_baseline' in locals() or 'f1_baseline' in globals():
    bilan_data.append({
        "Modèle": "Baseline",
        "ROI Std (%)": round(roi_std_bl, 2),
        "Mise Std": mise_std_bl,
        "Profit Std": round(profit_std_bl, 2),
        "ROI Sécu (%)": round(roi_sec_bl, 2),
        "Mise Sécu": mise_sec_bl,
        "Profit Sécu": round(profit_sec_bl, 2),
        "F1-Score (Seuil)": round(f1_baseline, 3)
    })

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]

    preds_std = (probas >= 0.50).astype(int)
    roi_std, mise_std, profit_std = compute_roi(y_test, preds_std, cotes_test_bilan)

    preds_sec = (probas >= SEUIL_SECURISE).astype(int)
    roi_sec, mise_sec, profit_sec = compute_roi(y_test, preds_sec, cotes_test_bilan)

    # recherche du Seuil Optimal
    meilleur_profit = -float('inf')
    meilleur_roi = -999
    meilleur_seuil = 0.50
    meilleure_mise = 0

    for seuil in seuils_a_tester:
        preds_temp = (probas >= seuil).astype(int)
        roi_temp, mise_temp, profit_temp = compute_roi(y_test, preds_temp, cotes_test_bilan)

        if profit_temp > meilleur_profit:
            meilleur_profit = profit_temp
            meilleur_roi = roi_temp
            meilleur_seuil = seuil
            meilleure_mise = mise_temp

    # Calcul du F1-Score associé à ce seuil dynamique optimal
    predictions_seuil_opti = (probas >= meilleur_seuil).astype(int)
    f1_opti = compute_f1(y_test, predictions_seuil_opti, TARGET_BIN)

    bilan_data.append({
        "Modèle": name,
        "ROI Std (%)": round(roi_std, 2),
        "Mise Std": mise_std,
        "Profit Std": round(profit_std, 2),
        "ROI Sécu (%)": round(roi_sec, 2),
        "Mise Sécu": mise_sec,
        "Profit Sécu": round(profit_sec, 2),
        "F1-Score (Seuil)": round(f1_opti, 3)  # F1 correspondant au seuil optimal trouvé
    })

    # Pour le dictionnaire de tuning dynamique
    tuning_resultats[name] = {
        "Profit Max": round(meilleur_profit, 1),
        "Seuil Optimal": round(meilleur_seuil, 2),
        "ROI (%)": round(meilleur_roi, 2),
        "Mise Totale": meilleure_mise,
        "F1 (Seuil)": round(f1_opti, 3)
    }

print("\n" + "=" * 95)
print(f" TABLEAU DE BORD FINANCIER STATIQUE (Seuil Fixe = {SEUIL_SECURISE})")
print("=" * 95)
df_bilan = pd.DataFrame(bilan_data)
if not df_bilan.empty:
    print(df_bilan.sort_values("F1-Score (Seuil)", ascending=False).to_string(index=False))
else:
    print("Aucune donnée disponible.")

print("\n" + "=" * 75)
print(" TUNING DES SEUILS (Dynamique)")
print("=" * 75)
print(pd.DataFrame.from_dict(tuning_resultats, orient='index').sort_values("ROI (%)", ascending=False).to_string())
print("=" * 75)

# %%list des seuils opti trouvé :
res_seuil_opti = {}
for nom, info in tuning_resultats.items():
    seuil = info['Seuil Optimal']
    res_seuil_opti[nom] = seuil
print("\n Seuils optimaux identifiés :")
for nom, seuil in res_seuil_opti.items():
    print(f" - {nom} :{seuil}")

#  Seuils optimaux identifiés :
#  - LogisticRegression (Base) :0.68
#  - LogisticRegression (Optimisé) :0.68
#  - RandomForest :0.81
#  - RandomForest (optimisé) :0.67
#  - XGBoost (Base) :0.75
#  - XGBoost (Optimisé) :0.66
#  - SVM RBF :0.68
#  - LightGBM :0.73
#  - LightGBM (Optimisé) :0.68

# %%=============================================================================
# VISUALISATION DES MÉTRIQUES DE PERFORMANCE
# =============================================================================

# On passe à 1 ligne et 2 colonnes, et on ajuste la largeur (13 au lieu de 18)
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].set_title("Courbes Precision-Recall (Rendement)")
axs[1].set_title("Courbes de Calibration (Fiabilité Probas)")

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]

    #  Courbe Precision-Recall : plus pertinente en cas de classes déséquilibrées 
    # Haute précision = peu de faux positifs (on ne mise pas sur des matchs perdus)
    # Haut rappel = on couvre un maximum de victoires réelles
    precision, recall, _ = precision_recall_curve(y_test, probas)
    axs[0].plot(recall, precision, label=name)

    #  Courbe de Calibration : évalue si les probabilités sont réalistes 
    # Un modèle parfaitement calibré suit la diagonale y=x
    # Brier Score : erreur quadratique moyenne des probabilités (plus bas = meilleur)
    brier = brier_score_loss(y_test, probas)
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=5, strategy='uniform')
    axs[1].plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier: {brier:.3f})")

#  Mise en forme des graphiques 

# Precision-Recall (désormais à l'index 0)
axs[0].set_xlabel('Recall (Couverture du marché)')
axs[0].set_ylabel('Precision (Fiabilité des picks)')
axs[0].legend(loc="lower left")

# Calibration (désormais à l'index 1)
axs[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Parfaite Calibration")
axs[1].set_xlabel('Probabilité prédite')
axs[1].set_ylabel('Fréquence réelle constatée')
axs[1].legend(loc="upper left")

plt.tight_layout()
plt.show()


# %% =========================================================================
# EVALUATION : COMPARAISON DES MISES FIXES VS MISES DYNAMIQUES
# =========================================================================

cotes_test_bilan = df[COTES_MAP[TARGET_BIN]].iloc[split_idx:].values
bilan_data_dynamique = []

# Baseline
preds_baseline = np.ones(len(y_test), dtype=int)
f1_baseline = compute_f1(y_test, preds_baseline, TARGET_BIN)
roi_bl, mise_bl, profit_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan, mise=10)

bilan_data_dynamique.append({
    "Modèle": "Baseline (Toujours 1)",
    "Seuil Opti": "-",
    #  Seuil fixe 0.50 
    "ROI Std 0.50 (%)": f"{roi_bl:.2f}",
    "Profit Std": f"{profit_bl:.1f}",
    "|": "|",
    #  Seuil sécurisé 0.65 
    "ROI Seuil opti (%)": f"{roi_sec_bl:.2f}",
    "Profit Sécu": f"{profit_sec_bl:.1f}",
    "mise Sécu": mise_sec_bl,
    "||": "||",
    #  Seuil optimal + paliers dynamiques 
    "ROI Dyn Opti (%)": f"{roi_bl:.2f}",
    "Profit Dyn": f"{profit_bl:.1f}",
    "mise Dyn": mise_bl,
})

seuils_modeles = {
    nom: info["Seuil Optimal"]
    for nom, info in tuning_resultats.items()
}

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]

    seuil_custom = seuils_modeles.get(name, SEUIL_SECURISE)

    # Seuil standard 0.50
    preds_std = (probas >= 0.50).astype(int)
    roi_std, mise_std, profit_std = compute_roi(y_test, preds_std, cotes_test_bilan, mise=MISE_UNITE)
    f1_std = compute_f1(y_test, preds_std, TARGET_BIN)

    # Seuil opti/modèles
    preds_sec = (probas >= seuil_custom).astype(int)
    roi_sec, mise_sec, profit_sec = compute_roi(y_test, preds_sec, cotes_test_bilan, mise=MISE_UNITE)
    f1_sec = compute_f1(y_test, preds_sec, TARGET_BIN)

    # Seuil opti+ stratégie dynamique 
    roi_dyn, mise_dyn, profit_dyn = compute_roi_par_modele(y_test, probas, cotes_test_bilan, seuil_custom)
    preds_opti = (probas >= seuil_custom).astype(int)
    f1_opti = compute_f1(y_test, preds_opti, TARGET_BIN)

    bilan_data_dynamique.append({
        "Modèle": name,
        "Seuil Opti": seuil_custom,
        #  Seuil fixe 0.50 
        "ROI Std 0.50 (%)": f"{roi_std:.2f}",
        "Profit Std": f"{profit_std:.1f}",
        "|": "|",
        #  Seuil sécurisé opti
        "ROI Seuil opti (%)": f"{roi_sec:.2f}",
        "Profit Sécu": f"{profit_sec:.1f}",
        "mise Sécu": mise_sec,
        "||": "||",
        #  Seuil optimal + paliers dynamiques 
        "ROI Dyn Opti (%)": f"{roi_dyn:.2f}",
        "Profit Dyn": f"{profit_dyn:.1f}",
        "mise Dyn": mise_dyn,
    })
# %%
print("\n" + "=" * 115)
print("             TABLEAU DE BORD COMPARATIF : MISES FIXES VS PALIERS DYNAMIQUES PAR MODÈLE")
print("=" * 115)
df_comparatif = pd.DataFrame(bilan_data_dynamique)
for c in ['ROI Std 0.50 (%)', 'Profit Std','ROI Seuil opti (%)', 'Profit Sécu', 'mise Sécu','ROI Dyn Opti (%)', 'Profit Dyn', 'mise Dyn']:
    df_comparatif[c] = pd.to_numeric(df_comparatif[c])

pd.set_option('display.expand_frame_repr', False)
print(df_comparatif.sort_values("ROI Dyn Opti (%)", ascending=False).to_string(index=False))
print("=" * 115)
# %%
# ===================================================================================================================
#              TABLEAU DE BORD COMPARATIF : MISES FIXES VS PALIERS DYNAMIQUES PAR MODÈLE
# ===================================================================================================================
#                        Modèle Seuil Opti ROI Std 0.50 (%) Profit Std | ROI Seuil opti (%) Profit Sécu  mise Sécu || ROI Dyn Opti (%) Profit Dyn  mise Dyn
#                       SVM RBF       0.68           -3.47%     -541.0 |              4.90%        20.6        420 ||            5.71%        9.7       170
#       RandomForest (optimisé)       0.67           -2.35%     -416.9 |              1.01%        70.6       7000 ||            0.94%       72.8      7720
#            XGBoost (Optimisé)       0.66           -1.92%     -339.8 |              1.29%        90.2       6970 ||            0.75%       52.1      6950
#     LogisticRegression (Base)       0.68           -3.21%     -572.4 |              1.49%       107.4       7230 ||            0.61%       51.5      8480
#                      LightGBM       0.73           -2.92%     -507.7 |              0.55%        26.1       4780 ||            0.44%       29.5      6670
#           LightGBM (Optimisé)       0.68           -1.99%     -351.9 |              0.36%        25.3       6950 ||            0.16%       14.2      8710
# LogisticRegression (Optimisé)       0.68           -2.84%     -498.8 |              0.96%        65.4       6800 ||           -1.03%      -78.6      7610
#                  RandomForest       0.81           -4.53%     -639.4 |              0.08%         1.0       1190 ||           -3.91%      -56.7      1450
#                XGBoost (Base)       0.75           -1.81%     -311.8 |              0.28%        11.1       3940 ||           -0.41%      -24.1      5810
#         Baseline (Toujours 1)          -           -6.57%    -2338.8 |             -1.13%       -72.4       6420 ||           -6.57%    -2338.8     35580
#                MLP (Optimisé)       0.63           -3.24%     -441.4 |              1.01%        71.1       7020 ||           -0.02%       -2.1      8800
#                    MLP (Base)       0.62           -4.42%     -580.4 |              0.81%        56.6       7030 ||           -1.57%     -144.6      9220
# ===================================================================================================================

# %%

from sklearn.metrics import log_loss, brier_score_loss, f1_score, precision_score, accuracy_score

bilan_metriques = []

# Baseline
probas_baseline_ml = np.full(len(y_test), y_train.mean())
bilan_metriques.append({
    "Modèle": "Baseline (Toujours 1)",
    "Log-Loss": round(log_loss(y_test, probas_baseline_ml), 4),
    "Brier Score": round(brier_score_loss(y_test, probas_baseline_ml), 4),
    "|": "|",
    "Accuracy (0.50)": round(accuracy_score(y_test, preds_baseline), 3),
    "F1 (0.50)": round(compute_f1(y_test, preds_baseline, TARGET_BIN), 3),
    "Précision (0.50)": round(precision_score(y_test, preds_baseline, zero_division=0), 3),
    "||": "||",
    "Seuil Opti": "-",
    "Accuracy (Opti)": round(accuracy_score(y_test, preds_baseline), 3),
    "F1 (Opti)": round(compute_f1(y_test, preds_baseline, TARGET_BIN), 3),
    "Précision (Opti)": round(precision_score(y_test, preds_baseline, zero_division=0), 3),
})

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]
    seuil_custom = seuils_modeles.get(name, SEUIL_SECURISE)

    preds_050 = (probas >= 0.50).astype(int)
    preds_opti = (probas >= seuil_custom).astype(int)

    bilan_metriques.append({
        "Modèle": name,
        "Log-Loss": round(log_loss(y_test, probas), 4),
        "Brier Score": round(brier_score_loss(y_test, probas), 4),
        "|": "|",
        "Accuracy (0.50)": round(accuracy_score(y_test, preds_050), 3),
        "F1 (0.50)": round(compute_f1(y_test, preds_050, TARGET_BIN), 3),
        "Précision (0.50)": round(precision_score(y_test, preds_050, zero_division=0), 3),
        "||": "||",
        "Seuil Opti": seuil_custom,
        "Accuracy (Opti)": round(accuracy_score(y_test, preds_opti), 3),
        "F1 (Opti)": round(compute_f1(y_test, preds_opti, TARGET_BIN), 3),
        "Précision (Opti)": round(precision_score(y_test, preds_opti, zero_division=0), 3),
    })
# %%
print("\n" + "=" * 130)
print("                         TABLEAU DE BORD COMPARATIF : MÉTRIQUES ML PAR MODÈLE")
print("=" * 130)
df_metriques = pd.DataFrame(bilan_metriques)
print(df_metriques.sort_values("Log-Loss", ascending=True).to_string(index=False))
print("=" * 130)

# %% sauvegarde des modèles

import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(scaler_mlp, "models/scaler_mlp.pkl")

for name, (model, X_te) in trained_models.items():
    nom_fichier = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"models/{nom_fichier}.pkl")
    print(f"Sauvegardé : models/{nom_fichier}.pkl")
