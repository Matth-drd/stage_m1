import sys
import os

# Ajoute le dossier contenant le script actuel au chemin de recherche
sys.path.append(os.path.abspath('code'))
import config as conf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import copy

import tqdm

print('Import')

# %%===========================================================================
# CONFIGURATION
# ===========================================================================

DATA_PATH = conf.path_clean_cell
SPLIT_RATIO = 0.8

features = conf.features
feat_imp = conf.features_importante
targets = conf.targets
bet = ['B365H', 'B365A', 'B365D']

# %%===========================================================================
# CHARGEMENT DES DONNÉES
# ===========================================================================

df = pd.read_csv(DATA_PATH)
df.sort_values("Date", inplace=True)

split = int(len(df) * SPLIT_RATIO)


# %%===========================================================================
# UTILITAIRES
# ===========================================================================

def _split(X, y, split):
    """Découpage train/test chronologique."""
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def best_target(scores):
    """Retourne la target avec le meilleur score."""
    idx = np.argmax(scores)
    return targets[idx], idx


def plot_confusion_matrix(y_true, y_pred, title, target_name, cmap="Blues"):
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)

    # On définit les étiquettes selon si c'est du multiclasse ou du binaire
    if target_name == "FTR":
        labels = ["Nul (0)", "Home (1)", "Away (2)"]
    else:
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


def plot_feature_importance(model, features_list: list, title):
    """Affiche les 15 features les plus importantes"""
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
    feat_importances.head(15).plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.ylabel("Score d'importance")
    plt.tight_layout()
    plt.show()
    return feat_importances.head(15)

def ROI(y_test, y_pred, cotes, mise=10):
    argent_mise = 0
    profit_net = 0
    y_test_vals = y_test.values
    cotes_vals = cotes.values

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            argent_mise += mise
            if y_test_vals[i] == 1:  # pari gagné
                profit_net += (mise * cotes_vals[i]) - mise
            else:
                profit_net -= mise

    if argent_mise == 0:
        return 0, 0, 0
    roi = (profit_net / argent_mise) * 100
    return roi, argent_mise, profit_net


# %%===========================================================================
# FONCTION PRINCIPALE DE PIPELINE
# ===========================================================================

def run_mod(df, feature_list, target_list, model_instance, verbose=False, seuil_conf=0.65):
    X_full = df[feature_list]
    results = []

    print(f"\n--- Entraînement : {type(model_instance).__name__} ---")

    for t in target_list:
        y = df[t]
        X_train, X_test, y_train, y_test = _split(X_full, y, split)

        # Gestion du Scaling
        if not isinstance(model_instance, RandomForestClassifier):
            scaler = StandardScaler()
            X_train_fit = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_list)
            X_test_fit = pd.DataFrame(scaler.transform(X_test), columns=feature_list)
        else:
            X_train_fit = X_train
            X_test_fit = X_test

        model_instance.fit(X_train_fit, y_train)
        y_pred = model_instance.predict(X_test_fit)

        avg = "weighted" if t == "FTR" else "binary"
        f1 = f1_score(y_test, y_pred, average=avg)

        results.append({'target': t, 'f1': f1})
        print(f"Target: {t} | F1: {f1:.3f}")

    #  best_targ
    best_res = max(results, key=lambda x: x['f1'])
    best_targ = best_res['target']
    best_f1 = best_res['f1']

    print(f"\n Meilleure target choisie: {best_targ} (F1: {best_f1:.3f})")

    # Ré-entraînement sur meilleure target
    y = df[best_targ]
    X_train, X_test, y_train, y_test = _split(X_full, y, split)

    if not isinstance(model_instance, RandomForestClassifier):
        scaler = StandardScaler()
        X_train_final = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_list)
        X_test_final = pd.DataFrame(scaler.transform(X_test), columns=feature_list)
    else:
        X_train_final = X_train
        X_test_final = X_test

    model_instance.fit(X_train_final, y_train)
    y_pred_final = model_instance.predict(X_test_final)

    # Affichage des rapports
    if verbose:
        print(f"\nRapport détaillé ({best_targ}) :")
        print(classification_report(y_test, y_pred_final))
        plot_confusion_matrix(y_test, y_pred_final, f"CM — {type(model_instance).__name__}", target_name=best_targ)

        if isinstance(model_instance, RandomForestClassifier):
            plot_feature_importance(model_instance, feature_list, f"Importance des variables — {best_targ}")

    # Calcul du ROI
    roi_final = 0
    cotes_map = {"Hvs": "B365H", "Avs": "B365A", "Dvs": "B365D"}

    if best_targ in cotes_map:
        col_cote = cotes_map[best_targ]
        cotes_test = df[col_cote].iloc[split:]

        # On utilise predict_proba pour filtrer avec le seuil
        y_proba = model_instance.predict_proba(X_test_final)[:, 1]
        y_pred_custom = (y_proba >= seuil_conf).astype(int)

        roi_final, mise, benef = ROI(y_test, pd.Series(y_pred_custom), cotes_test, mise=10)

        status = "gain net" if benef >= 0 else "perte net"
        print(f"ROI ({seuil_conf}): {roi_final:.2f}% | Mise totale: {mise}€ | {status}: {benef:.1f}€")
    else:
        print(f"Calcul du ROI non configuré pour la target {best_targ}")

    return model_instance, best_targ, best_f1, roi_final


# %%===========================================================================
# EXÉCUTION EN BOUCLE ET BILAN FINAL
# ===========================================================================

feature_sets = {
    "All Features": features,
    "Important Features": feat_imp
}

models_to_test = [
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    SVC(probability=True, kernel='rbf', class_weight="balanced", random_state=42),
    KNeighborsClassifier(n_neighbors=10),
]

historique_bilan = []

for label, f_list in tqdm(feature_sets.items()):
    print(f"\n{'#' * 40}\n TESTING FEATURE SET : {label}\n{'#' * 40}")
    for m in tqdm(models_to_test):
        # Copie profonde pour repartir d'un modèle vierge
        fresh_model = copy.deepcopy(m)
        mod, targ, f1_score_val, roi_val = run_mod(df, f_list, targets, fresh_model, verbose=True, seuil_conf=0.65)

        historique_bilan.append({
            "Feature_Set": label,
            "Model": type(m).__name__,
            "Best_Target": targ,
            "F1_Score": round(f1_score_val, 3),
            "ROI_(%)": round(roi_val, 2)
        })

# Affichage du DataFrame final
df_bilan = pd.DataFrame(historique_bilan)
print("\n" + "=" * 55)
print(" RECAP des Perfs (Trié par ROI)")
print("=" * 55)
print(df_bilan.sort_values(by="ROI_(%)", ascending=False).to_string(index=False))
