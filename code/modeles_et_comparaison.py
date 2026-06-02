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
import scikit_posthocs as sp
import seaborn as sns
import umap
from scipy.stats import friedmanchisquare, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, brier_score_loss,
    classification_report, f1_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

matplotlib.use('TkAgg')

sys.path.append(os.path.abspath('code'))
import config as conf

print("Import")

# %%=============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

DATA_PATH = conf.path_clean_cell
SPLIT_RATIO = 0.8
SEUIL_SECURISE = 0.65
MISE_UNITE = 10

COTES_MAP = {"Hvs": "BWH", "Avs": "BWA", "Dvs": "BWD"}

FEATURES = conf.ft_commune
TARGETS = conf.targets
TARGET_BIN = "Hvs"


# %%=============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def split_data(X, y, ratio=SPLIT_RATIO):
    """
    Découpe les données en ensembles d'entraînement et de test en respectant
    l'ordre chronologique

    Parameters
    ----------
    X : pd.DataFrame
        Matrice de features.
    y : pd.Series
        Vecteur cible.
    ratio : float, optional
        Proportion des données allouées à l'entraînement. Par défaut : 0.8.
    Returns
    -------
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
    ----------
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.
    target_name : str
        Nom de la variable cible.

    Returns
    -------
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
    ----------
    y_test : array-like
        Résultats réels des matchs (1 = victoire, 0 = autre).
    y_pred : array-like
        Prédictions du modèle (1 = pari placé, 0 = pas de pari).
    cotes : array-like
        Cotes bookmaker associées à chaque match.
    mise : float, optional
        Montant misé par pari. Par défaut : MISE_UNITE (10€).

    Returns
    -------
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


# %%=============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

X_brut = df[FEATURES]

split_idx = int(len(df) * SPLIT_RATIO)

y_train_fin = df[TARGET_BIN].iloc[:split_idx]
ratio_poids = (
    (y_train_fin == 0).sum() / (y_train_fin == 1).sum()
    if (y_train_fin == 1).sum() > 0
    else 1.0
)

# %%=============================================================================
# DÉFINITION DES MODÈLES ET GRILLES D'HYPERPARAMÈTRES
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

    }
}
# %%=============================================================================
# ENTRAÎNEMENT ET OPTIMISATION DES MODÈLES
# =============================================================================

print("\n" + "=" * 70)
print(" train et opti")
print("=" * 70)

scaler = StandardScaler()
X_train_brut, X_test_brut, y_train, y_test = split_data(X_brut, df[TARGET_BIN])

X_train_scaled = scaler.fit_transform(X_train_brut)
X_test_scaled = scaler.transform(X_test_brut)

# Stockage des modèles entraînés et de leurs données de test associées
trained_models = {}

for name, config in models_pipeline.items():
    X_tr = X_train_scaled if config["scaling"] else X_train_brut.values
    X_te = X_test_scaled if config["scaling"] else X_test_brut.values

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
roi_std_bl, mise_std_bl, profit_std_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan)
roi_sec_bl, mise_sec_bl, profit_sec_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan)
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
print(f"\n BASELINE — Accuracy : {y_test.mean():.1%} | F1 : {f1_baseline:.3f} | ROI : {roi_std_bl:.2f}%")

# =====================================
# %%=====================================

# Initialisation des structures de stockage
bilan_data = []
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
for nom, info in tuning_resultats.items() :
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

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].set_title("Courbes ROC (Discrimination)")
axs[1].set_title("Courbes Precision-Recall (Rendement)")
axs[2].set_title("Courbes de Calibration (Fiabilité Probas)")

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]

    # --- Courbe ROC : mesure la capacité du modèle à discriminer les classes ---
    # AUC proche de 1 = excellent modèle, proche de 0.5 = équivalent au hasard
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # --- Courbe Precision-Recall : plus pertinente en cas de classes déséquilibrées ---
    # Haute précision = peu de faux positifs (on ne mise pas sur des matchs perdus)
    # Haut rappel = on couvre un maximum de victoires réelles
    precision, recall, _ = precision_recall_curve(y_test, probas)
    axs[1].plot(recall, precision, label=name)

    # --- Courbe de Calibration : évalue si les probabilités sont réalistes ---
    # Un modèle parfaitement calibré suit la diagonale y=x
    # Brier Score : erreur quadratique moyenne des probabilités (plus bas = meilleur)
    brier = brier_score_loss(y_test, probas)
    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=5, strategy='uniform')
    axs[2].plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier: {brier:.3f})")

# --- Mise en forme des graphiques ---

# ROC : la diagonale représente un classifieur aléatoire (baseline)
axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axs[0].set_xlabel('Taux de Faux Positifs')
axs[0].set_ylabel('Taux de Vrais Positifs')
axs[0].legend(loc="lower right")

# Precision-Recall
axs[1].set_xlabel('Recall (Couverture du marché)')
axs[1].set_ylabel('Precision (Fiabilité des picks)')
axs[1].legend(loc="lower left")

# Calibration : la diagonale représente une calibration parfaite
axs[2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Parfaite Calibration")
axs[2].set_xlabel('Probabilité prédite')
axs[2].set_ylabel('Fréquence réelle constatée')
axs[2].legend(loc="upper left")

plt.tight_layout()
plt.show()

# %% sauvegarde des modèles

# import joblib
# import os
#
# os.makedirs("models", exist_ok=True)
# joblib.dump(scaler, "models/scaler.pkl")
#
# for name, (model, X_te) in trained_models.items():
#     nom_fichier = name.replace(" ", "_").replace("(", "").replace(")", "")
#     joblib.dump(model, f"models/{nom_fichier}.pkl")
#     print(f"Sauvegardé : models/{nom_fichier}.pkl")
