"""
=============================================================================
PIPELINE DE PRÉDICTION DE MATCHS DE FOOTBALL — MODÈLES DE CLASSIFICATION
=============================================================================
Ce script entraîne et évalue plusieurs modèles de machine learning pour prédire
l'issue binaire d'un match de football (victoire à domicile ou non).
"""

import os
import sys

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.append(os.path.abspath("code"))
import config as conf

# %%=============================================================================
# CONFIGURATION GÉNÉRALE
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
    """Découpe X et y en train/test selon un ratio chronologique."""
    sp = int(len(X) * ratio)
    return X.iloc[:sp], X.iloc[sp:], y.iloc[:sp], y.iloc[sp:]


def compute_f1(y_true, y_pred, target_name):
    """Calcule le F1-score (pondéré pour FTR, binaire sinon)."""
    avg = "weighted" if target_name == "FTR" else "binary"
    return f1_score(y_true, y_pred, average=avg, zero_division=0)


def compute_roi(y_test, y_pred, cotes, mise=MISE_UNITE):
    """ROI pour une stratégie de mise fixe sur chaque pari prédit positif."""
    y_t = np.array(y_test)
    y_p = np.array(y_pred)
    c = np.array(cotes)

    mask = y_p == 1
    total_mise = mask.sum() * mise

    if total_mise == 0:
        return 0.0, 0, 0.0

    gains = np.where(y_t[mask] == 1, mise * c[mask], 0.0)
    profit_net = gains.sum() - total_mise
    roi = (profit_net / total_mise) * 100

    return roi, int(total_mise), profit_net


def compute_roi_par_modele(y_test, probas, cotes, seuil_optimal):
    """ROI avec stratégie de mises par paliers selon la confiance du modèle."""
    y_t = np.array(y_test)
    c = np.array(cotes)
    p = np.array(probas)

    prob_bookmaker = 1 / c
    is_value = p > prob_bookmaker
    mises = np.zeros(len(p))

    mises[(p >= seuil_optimal) & (p < seuil_optimal + 0.10) & is_value] = 10.0
    mises[(p >= seuil_optimal + 0.10) & (p <= 1.0) & is_value] = 30.0

    mask = mises > 0
    total_mise = mises[mask].sum()

    if total_mise == 0:
        return 0.0, 0, 0.0

    gains = np.where(y_t[mask] == 1, mises[mask] * c[mask], 0.0)
    profit_net = gains.sum() - total_mise
    roi = (profit_net / total_mise) * 100

    return roi, int(total_mise), profit_net


# %%=============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

X_brut = df[FEATURES]
X_brut_nn = df[conf.ft_nn]

split_idx = int(len(df) * SPLIT_RATIO)
y_train_fin = df[TARGET_BIN].iloc[:split_idx]

ratio_poids = (
    (y_train_fin == 0).sum() / (y_train_fin == 1).sum()
    if (y_train_fin == 1).sum() > 0
    else 1.0
)

# %%=============================================================================
# GRILLES DE RECHERCHE D'HYPERPARAMÈTRES
# =============================================================================

cv_tempo = TimeSeriesSplit(n_splits=3)

lr_grid = {
    "penalty": ["l1", "l2"],
    "C": np.logspace(-3, 3, 20),
    "solver": ["saga"],
    "tol": [1e-4, 1e-3, 1e-2],
}

xgb_grid = {
    "n_estimators": randint(50, 350),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": [3, 4, 5, 6],
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(1, 5),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}

lgbm_grid = {
    "n_estimators": randint(50, 350),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": [3, 4, 5, 6],
    "num_leaves": randint(20, 100),
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(1, 5),
}

rf_grid = {
    "n_estimators": randint(50, 350),
    "max_depth": [3, 4, 5, 6],
}

mlp_grid = {
    "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (100, 100)],
    "alpha": uniform(1e-5, 1e-1),
    "learning_rate_init": uniform(1e-4, 1e-2),
    "batch_size": [32, 64, 128],
}

svm_grid = {
    "C": np.logspace(-2, 2, 10),
    "gamma": ["scale", "auto"] + list(np.logspace(-3, -1, 5)),
}

# %%=============================================================================
# PIPELINE DES MODÈLES
# =============================================================================

models_pipeline = {
    "LogisticRegression (Base)": {
        "instance": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "scaling": True,
        "optimize": False,
    },
    "LogisticRegression (Optimisé)": {
        "instance": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
        "scaling": True,
        "optimize": True,
        "search_args": {
            "param_distributions": lr_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
    "RandomForest": {
        "instance": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "scaling": False,
        "optimize": False,
    },
    "RandomForest (optimisé)": {
        "instance": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": rf_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
    "XGBoost (Base)": {
        "instance": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=ratio_poids,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "scaling": False,
        "optimize": False,
    },
    "XGBoost (Optimisé)": {
        "instance": XGBClassifier(
            scale_pos_weight=ratio_poids,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": xgb_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
    "SVM RBF (base)": {
        "instance": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "scaling": True,
        "optimize": False,
    },
    "SVM RBF (Optimisé)": {
        "instance": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "scaling": True,
        "optimize": True,
        "search_args": {
            "param_distributions": svm_grid,
            "n_iter": 10,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
    "LightGBM (Base)": {
        "instance": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "scaling": False,
        "optimize": False,
    },
    "LightGBM (Optimisé)": {
        "instance": lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "scaling": False,
        "optimize": True,
        "search_args": {
            "param_distributions": lgbm_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
    "MLP (Base)": {
        "instance": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=800,
            alpha=0.0001,
            solver="adam",
            random_state=21,
            tol=1e-9,
        ),
        "scaling": True,
        "nn": True,
        "optimize": False,
    },
    "MLP (Optimisé)": {
        "instance": MLPClassifier(max_iter=800, solver="adam", random_state=42, tol=1e-9),
        "scaling": True,
        "nn": True,
        "optimize": True,
        "search_args": {
            "param_distributions": mlp_grid,
            "n_iter": 15,
            "scoring": "neg_log_loss",
            "cv": cv_tempo,
        },
    },
}

# %%=============================================================================
# ENTRAÎNEMENT ET RECHERCHE D'HYPERPARAMÈTRES
# =============================================================================

print("\n" + "=" * 70)
print(" ENTRAÎNEMENT ET OPTIMISATION DES MODÈLES")
print("=" * 70)

scaler = StandardScaler()
X_train_brut, X_test_brut, y_train, y_test = split_data(X_brut, df[TARGET_BIN])

X_train_scaled = scaler.fit_transform(X_train_brut)
X_test_scaled = scaler.transform(X_test_brut)

scaler_mlp = StandardScaler()
X_train_brut_nn, X_test_brut_nn, _, _ = split_data(X_brut_nn, df[TARGET_BIN])
X_train_scaled_nn = scaler_mlp.fit_transform(X_train_brut_nn)
X_test_scaled_nn = scaler_mlp.transform(X_test_brut_nn)

trained_models = {}

for name, cfg in tqdm(models_pipeline.items(), total=len(models_pipeline)):
    is_nn = cfg.get("nn", False)

    if is_nn:
        X_tr, X_te = X_train_scaled_nn, X_test_scaled_nn
    elif cfg["scaling"]:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train_brut, X_test_brut

    if cfg["optimize"]:
        tqdm.write(f" Optimisation RandomSearch pour : {name}")
        search = RandomizedSearchCV(
            estimator=cfg["instance"],
            **cfg["search_args"],
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_tr, y_train)
        tqdm.write(f" -> Meilleurs paramètres : {search.best_params_}")
        trained_models[name] = (search.best_estimator_, X_te)
    else:
        tqdm.write(f" Entraînement de base pour : {name}")
        model = cfg["instance"].fit(X_tr, y_train)
        trained_models[name] = (model, X_te)

# %%=============================================================================
# OPTIMISATION DES SEUILS ET SIMULATION ROI
# =============================================================================

cotes_test_bilan = df[COTES_MAP[TARGET_BIN]].iloc[split_idx:]
seuils_a_tester = np.arange(0.40, 0.85, 0.01)
bilan_data = []

# --- Baseline ---
preds_baseline = np.ones(len(y_test), dtype=int)
f1_baseline = compute_f1(y_test, preds_baseline, TARGET_BIN)
probas_baseline = 1 / cotes_test_bilan
preds_baseline_sec = (probas_baseline >= SEUIL_SECURISE).astype(int)

roi_std_bl, mise_std_bl, profit_std_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan)
roi_sec_bl, mise_sec_bl, profit_sec_bl = compute_roi(y_test, preds_baseline_sec, cotes_test_bilan)

bilan_data.append({
    "Modèle": "Baseline",
    "ROI Std (%)": round(roi_std_bl, 2),
    "Mise Std": mise_std_bl,
    "Profit Std": round(profit_std_bl, 2),
    "|": "|",
    "ROI Sécu (%)": round(roi_sec_bl, 2),
    "Mise Sécu": mise_sec_bl,
    "Profit Sécu": round(profit_sec_bl, 2),
    "F1-Score (Seuil)": round(f1_baseline, 3),
})

tuning_resultats = {}

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]

    preds_std = (probas >= 0.50).astype(int)
    roi_std, mise_std, profit_std = compute_roi(y_test, preds_std, cotes_test_bilan)

    preds_sec = (probas >= SEUIL_SECURISE).astype(int)
    roi_sec, mise_sec, profit_sec = compute_roi(y_test, preds_sec, cotes_test_bilan)

    meilleur_profit = -float("inf")
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

    predictions_seuil_opti = (probas >= meilleur_seuil).astype(int)
    f1_opti = compute_f1(y_test, predictions_seuil_opti, TARGET_BIN)

    bilan_data.append({
        "Modèle": name,
        "ROI Std (%)": round(roi_std, 2),
        "Mise Std": mise_std,
        "Profit Std": round(profit_std, 2),
        "|": "|",
        "ROI Sécu (%)": round(roi_sec, 2),
        "Mise Sécu": mise_sec,
        "Profit Sécu": round(profit_sec, 2),
        "F1-Score (Seuil)": round(f1_opti, 3),
    })

    tuning_resultats[name] = {
        "Profit Max": round(meilleur_profit, 1),
        "Seuil Optimal": round(meilleur_seuil, 2),
        "ROI (%)": round(meilleur_roi, 2),
        "Mise Totale": meilleure_mise,
        "F1 (Seuil)": round(f1_opti, 3),
    }

# %%--- Affichage des tableaux ---
print("\n" + "=" * 95)
print(f" TABLEAU DE BORD FINANCIER STATIQUE (Seuil Fixe = {SEUIL_SECURISE})")
print("=" * 95)
print(pd.DataFrame(bilan_data).sort_values("F1-Score (Seuil)", ascending=False).to_string(index=False))

print("\n" + "=" * 75)
print(" TUNING DES SEUILS (Dynamique)")
print("=" * 75)
print(pd.DataFrame.from_dict(tuning_resultats, orient="index").sort_values("ROI (%)", ascending=False).to_string())

seuils_modeles = {nom: info["Seuil Optimal"] for nom, info in tuning_resultats.items()}

# %%=============================================================================
# EVALUATION GLOBALE ET STRATÉGIE PAR PALIERS DYNAMIQUES
# =============================================================================

cotes_test_bilan_vals = cotes_test_bilan.values
bilan_data_dynamique = []

# Baseline Dynamique
roi_bl, mise_bl, profit_bl = compute_roi(y_test, preds_baseline, cotes_test_bilan_vals, mise=10)
bilan_data_dynamique.append({
    "Modèle": "Baseline (Toujours 1)",
    "Seuil Opti": "-",
    "ROI Std 0.50 (%)": roi_bl,
    "Profit Std": profit_bl,
    "|": "|",
    "ROI Seuil opti (%)": roi_sec_bl,
    "Profit Sécu": profit_sec_bl,
    "mise Sécu": mise_sec_bl,
    "||": "||",
    "ROI Dyn Opti (%)": roi_bl,
    "Profit Dyn": profit_bl,
    "mise Dyn": mise_bl,
})

for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]
    seuil_custom = seuils_modeles.get(name, SEUIL_SECURISE)

    preds_std = (probas >= 0.50).astype(int)
    roi_std, mise_std, profit_std = compute_roi(y_test, preds_std, cotes_test_bilan_vals, mise=MISE_UNITE)

    preds_sec = (probas >= seuil_custom).astype(int)
    roi_sec, mise_sec, profit_sec = compute_roi(y_test, preds_sec, cotes_test_bilan_vals, mise=MISE_UNITE)

    roi_dyn, mise_dyn, profit_dyn = compute_roi_par_modele(y_test, probas, cotes_test_bilan_vals, seuil_custom)

    bilan_data_dynamique.append({
        "Modèle": name,
        "Seuil Opti": seuil_custom,
        "ROI Std 0.50 (%)": roi_std,
        "Profit Std": profit_std,
        "|": "|",
        "ROI Seuil opti (%)": roi_sec,
        "Profit Sécu": profit_sec,
        "mise Sécu": mise_sec,
        "||": "||",
        "ROI Dyn Opti (%)": roi_dyn,
        "Profit Dyn": profit_dyn,
        "mise Dyn": mise_dyn,
    })

# %%--- Affichage du Tableau Comparatif Financier ---
print("\n" + "=" * 115)
print("             TABLEAU DE BORD COMPARATIF : MISES FIXES VS PALIERS DYNAMIQUES PAR MODÈLE")
print("=" * 115)

df_comparatif = pd.DataFrame(bilan_data_dynamique)
colonnes_num = [
    "ROI Std 0.50 (%)", "Profit Std", "ROI Seuil opti (%)", "Profit Sécu",
    "mise Sécu", "ROI Dyn Opti (%)", "Profit Dyn", "mise Dyn", "Seuil Opti",
]
for col in colonnes_num:
    df_comparatif[col] = pd.to_numeric(df_comparatif[col], errors="coerce")

df_comparatif_trie = df_comparatif.sort_values("ROI Dyn Opti (%)", ascending=False).copy()
df_comparatif_trie["Seuil Opti"] = df_comparatif_trie["Seuil Opti"].fillna("-")

pd.set_option("display.expand_frame_repr", False)
print(df_comparatif_trie.to_string(index=False))
print("=" * 115)

# %%=============================================================================
# MÉTRIQUES MACHINE LEARNING
# =============================================================================

bilan_metriques = []
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

# %%=============================================================================
# SAUVEGARDE DES MODÈLES
# =============================================================================

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler_standard.pkl")
joblib.dump(scaler_mlp, "models/scaler_mlp.pkl")

for name, (model, _) in trained_models.items():
    nom_fichier = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"models/{nom_fichier}.pkl")
    print(f"Sauvegardé : models/{nom_fichier}.pkl")

# %%=============================================================================
# VISUALISATIONS
# =============================================================================

# --- Log-Loss par modèle ---
plt.figure()
plt.bar(df_metriques["Modèle"], df_metriques["Log-Loss"], edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Modèles")
plt.ylabel("Log-Loss")
plt.title("Comparaison du Log-Loss par Modèle")
plt.tight_layout()
plt.show()

# --- Métriques ML groupées ---
metriques_a_afficher = [
    "Accuracy (0.50)", "Accuracy (Opti)",
    "F1 (0.50)", "F1 (Opti)",
    "Précision (0.50)", "Précision (Opti)",
]
couleurs_ml = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#ff7f0e", "#ffbb78"]

modeles = df_metriques["Modèle"].tolist()
scores_metriques = {m: tuple(df_metriques[m].round(2)) for m in metriques_a_afficher}
x = np.arange(len(modeles))
width = 0.14

fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
for i, (nom_metrique, valeurs) in enumerate(scores_metriques.items()):
    rects = ax.bar(x + width * i, valeurs, width, label=nom_metrique, color=couleurs_ml[i], edgecolor="black")
    ax.bar_label(rects, padding=3, rotation=90, fontsize=8)

ax.set_title("Comparaison des performances par Modèle")
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(modeles, rotation=45, ha="right")
ax.legend(loc="upper left", ncols=3)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.show()

# --- ROI et Profits financiers ---
metriques_roi = ["ROI Std 0.50 (%)", "ROI Seuil opti (%)", "ROI Dyn Opti (%)"]
metriques_profit = ["Profit Std", "Profit Sécu", "Profit Dyn"]
couleurs_roi = ["#1d3557", "#457b9d", "#a8dadc"]
couleurs_profit = ["#ca6702", "#ee9b00", "#e9d8a6"]

df_graphique = df_comparatif_trie[df_comparatif_trie["Modèle"] != "Baseline (Toujours 1)"].copy()
modeles = df_graphique["Modèle"].tolist()
x = np.arange(len(modeles))
width = 0.25

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, layout="constrained")

for i, (metrique, c) in enumerate(zip(metriques_roi, couleurs_roi)):
    valeurs = df_graphique[metrique].round(2).tolist()
    rects = ax1.bar(x + width * i, valeurs, width, label=metrique, color=c, edgecolor="black")
    ax1.bar_label(rects, padding=3, rotation=90, fontsize=8, fmt="%.1f%%")

ax1.set_title("Comparaison du Retour sur Investissement (ROI)", fontsize=13, fontweight="bold")
ax1.set_ylabel("ROI (%)")
ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="lower left", ncols=3, fontsize=9)

for i, (metrique, c) in enumerate(zip(metriques_profit, couleurs_profit)):
    valeurs = df_graphique[metrique].round(2).tolist()
    rects = ax2.bar(x + width * i, valeurs, width, label=metrique, color=c, edgecolor="black")
    ax2.bar_label(rects, padding=3, rotation=90, fontsize=8, fmt="%.1f€")

ax2.set_title("Comparaison des Profits Nets", fontsize=13, fontweight="bold")
ax2.set_ylabel("Profit (€)")
ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="lower left", ncols=3, fontsize=9)
ax2.set_xticks(x + width)
ax2.set_xticklabels(modeles, rotation=45, ha="right")

plt.suptitle("Performances Financières des Modèles (Hors Baseline)", fontsize=16, fontweight="bold")
plt.show()

# %%=============================================================================
# PROFITS CUMULÉS EN SÉRIE TEMPORELLE (MATCH PAR MATCH)
# =============================================================================

y_test_arr = y_test
cotes_arr = cotes_test_bilan_vals
index_matchs = np.arange(1, len(y_test_arr) + 1)

dict_profits_cumules = {}

# Baseline
gains_bl = np.where(y_test_arr == 1, 10 * cotes_arr, 0.0)
dict_profits_cumules["Baseline (Toujours 1)"] = np.cumsum(gains_bl - 10)

# Modèles
for name, (model, X_te) in trained_models.items():
    probas = model.predict_proba(X_te)[:, 1]
    seuil_custom = seuils_modeles.get(name, SEUIL_SECURISE)

    # Standard (seuil 0.50, mise fixe 10€)
    preds_std = (probas >= 0.50).astype(int)
    gains_std = np.where((preds_std == 1) & (y_test_arr == 1), 10 * cotes_arr, 0.0)
    mises_std = np.where(preds_std == 1, 10, 0.0)
    dict_profits_cumules[f"{name} (Base)"] = np.cumsum(gains_std - mises_std)

    # Dynamique (mises par paliers)
    prob_bookmaker = 1 / cotes_arr
    is_value = probas > prob_bookmaker
    mises_dyn = np.zeros(len(probas))
    mises_dyn[(probas >= seuil_custom) & (probas < seuil_custom + 0.10) & is_value] = 10.0
    mises_dyn[(probas >= seuil_custom + 0.10) & (probas <= 1.0) & is_value] = 30.0

    gains_dyn = np.where((mises_dyn > 0) & (y_test_arr == 1), mises_dyn * cotes_arr, 0.0)
    dict_profits_cumules[f"{name} (Optimisé)"] = np.cumsum(gains_dyn - mises_dyn)

fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax, type_modele in zip(axs, ["Base", "Optimisé"]):
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.4)
    for nom, profit_serie in dict_profits_cumules.items():
        est_opti = "Optimisé" in nom
        is_baseline = nom.startswith("Baseline")
        show = (
                is_baseline
                or (type_modele == "Optimisé" and est_opti)
                or (type_modele == "Base" and not est_opti)
        )
        if show:
            style = {"linestyle": "--", "linewidth": 2.0, "color": "red"} if is_baseline else {"linewidth": 1.5}
            label = f"{nom.replace(' (Base)', '').replace(' (Optimisé)', '')} ({profit_serie[-1]:.1f}€)"
            ax.plot(index_matchs, profit_serie, label=label, **style)

    ax.set_title(f"Modèles : {type_modele}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Matchs Évalués (Chronologique)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, ncols=2)

axs[0].set_ylabel("Profit Net Cumulé (€)", fontsize=11)
plt.tight_layout()
plt.show()

# %%
