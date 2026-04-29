"""Analyse et comparaison de modèles de classification sur les résultats de matchs.

Pipeline :
    1. Baseline — Régression Logistique (80/20 chronologique)
    2. Random Forest — découpage unique puis validation croisée temporelle
    3. Sélection automatique de la meilleure target via F1-score
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# %%===========================================================================
# CONFIGURATION
# ===========================================================================

DATA_PATH = "data/csv/foot_v4.csv"
SPLIT_RATIO = 0.8

FEATURES_FT = [
    "FT_Hforme", "FT_Hatt", "FT_Hdef",
    "FT_Aforme", "FT_Aatt", "FT_Adef",
    "FT_Hprecision", "FT_Aprecision",
    "FT_Hprec_weight", "FT_Aprec_weight",
    "FT_Elo_H", "FT_Elo_A", "FT_Elo_dif",
]

FEATURES_HT = [
    "HT_Hforme", "HT_Hdef",
    "HT_Aforme", "HT_Adef",
]

FEATURES = FEATURES_FT + FEATURES_HT
TARGETS = ["FTR", "Hvs", "Avs", "Dvs"]

# %%===========================================================================
# CHARGEMENT DES DONNÉES
# ===========================================================================

df = pd.read_csv(DATA_PATH)
df.sort_values("Date", inplace=True)

df = df[FEATURES + TARGETS].copy()
X = df[FEATURES]

split = int(len(df) * SPLIT_RATIO)


# %%===========================================================================
# UTILITAIRES
# ===========================================================================

def chronological_split(X, y, split):
    """Découpage train/test chronologique."""
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def best_target(scores: dict) -> str:
    """Retourne la target avec le meilleur score."""
    return max(scores, key=scores.get)


def plot_confusion_matrix(y_true, y_pred, title: str, target_name: str, cmap: str = "Blues"):
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)

    # On définit les étiquettes selon si c'est du multiclasse ou du binaire
    if target_name == "FTR":
        labels = ["Nul (0)", "Home (1)", "Away (2)"]
    else:
        # Pour Hvs, Avs, Dvs, le 0 veut dire "Autre" et le 1 veut dire "Succès"
        labels = [f"Non {target_name}", target_name]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Prédictions")
    plt.ylabel("Réalité")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, features: list, title: str):
    plt.figure(figsize=(10, 6))
    feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    feat_imp.plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.ylabel("Score d'importance")
    plt.tight_layout()
    plt.show()


# %%===========================================================================
# BASELINE — RÉGRESSION LOGISTIQUE
# ===========================================================================
print("=" * 50)
print("BASELINE — RÉGRESSION LOGISTIQUE")
print("=" * 50)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
scaler = StandardScaler()

lr_scores = {}
for targ in TARGETS:
    y = df[targ]
    X_train, X_test, y_train, y_test = chronological_split(X, y, split)

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr.fit(X_train_sc, y_train)
    y_pred = lr.predict(X_test_sc)

    if targ == "FTR":
        f1 = f1_score(y_test, y_pred, average="weighted")
    else:
        f1 = f1_score(y_test, y_pred, average="binary")

    lr_scores[targ] = f1
    print(
        f"[{targ}] Train: {lr.score(X_train_sc, y_train):.3f} | Test: {lr.score(X_test_sc, y_test):.3f} | F1: {f1:.3f}")

best_targ_lr = best_target(lr_scores)
print(f"\n→ Meilleure target LR : {best_targ_lr} (F1 = {lr_scores[best_targ_lr]:.3f})")

# Rapport détaillé sur la meilleure target
y = df[best_targ_lr]
X_train, X_test, y_train, y_test = chronological_split(X, y, split)
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
lr.fit(X_train_sc, y_train)
print(f"\nRapport détaillé — target : {best_targ_lr}")
print(classification_report(y_test, lr.predict(X_test_sc)))

# %%===========================================================================
# RANDOM FOREST — DÉCOUPAGE UNIQUE
# ===========================================================================

print("=" * 50)
print("RANDOM FOREST — DÉCOUPAGE UNIQUE (80/20)")
print("=" * 50)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

rf_scores = {}
for targ in TARGETS:
    y = df[targ]
    X_train, X_test, y_train, y_test = chronological_split(X, y, split)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    if targ == "FTR":
        f1 = f1_score(y_test, y_pred, average="weighted")
    else:
        f1 = f1_score(y_test, y_pred, average="binary")

    rf_scores[targ] = f1
    print(f"[{targ}] F1-score: {f1:.3f}")

best_targ_rf = best_target(rf_scores)
print(f"\n→ Meilleure target RF : {best_targ_rf} (Accuracy = {rf_scores[best_targ_rf]:.3f})")

# Rapport + graphiques sur la meilleure target
y = df[best_targ_rf]
X_train, X_test, y_train, y_test = chronological_split(X, y, split)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"\nRapport détaillé — target : {best_targ_rf}")
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix — RF — {best_targ_rf}", cmap="Greens",
                      target_name=best_targ_rf)
plot_feature_importance(rf, FEATURES, f"Importance des variables — RF — {best_targ_rf}")

# %%===========================================================================
# RANDOM FOREST — VALIDATION CROISÉE TEMPORELLE
# ===========================================================================

print("=" * 50)
print("RANDOM FOREST — VALIDATION CROISÉE TEMPORELLE")
print("=" * 50)

y = df[best_targ_rf]
tscv = TimeSeriesSplit(n_splits=5)

for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    rf.fit(X_tr, y_tr)
    if best_targ_rf == "FTR":
        f1 = f1_score(y_test, y_pred, average="weighted")
    else:
        f1 = f1_score(y_test, y_pred, average="binary")
    print(f"Fold {i + 1} — F1-score : {f1:.3f}")

# Rapport sur le dernier fold (le plus récent)
y_pred_cv = rf.predict(X_te)
print(f"\nRapport détaillé — dernier fold — target : {best_targ_rf}")
print(classification_report(y_te, y_pred_cv))

plot_confusion_matrix(y_te, y_pred_cv, f"Confusion Matrix — RF CV (dernier fold) — {best_targ_rf}",
                      target_name=best_targ_rf)
