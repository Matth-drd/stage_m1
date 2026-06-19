import sys
import os

sys.path.append(os.path.abspath('code'))
import config as conf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score

# %%
df = pd.read_csv(conf.path_clean_cell)
features = conf.features
y = df["Hvs"]
X = df[features]
split = int(len(df) * conf.split_ratio)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% RFECV
estimator = LogisticRegression(max_iter=1_000)
rfecv = RFECV(estimator, step=1, cv=5, scoring='f1')
rfecv.fit(X_scaled, y)

features_rfecv = pd.Series(features)[rfecv.support_]
print(f"Nombre optimal de features (RFECV) : {rfecv.n_features_}")
print(f"Features RFECV : {features_rfecv.values}")
# Nombre optimal de features (RFECV) : 34
# Features RFECV : ['FT_Hforme' 'FT_Aforme' 'FT_Aatt' 'FT_Hdef' 'FT_Adef' 'FT_forme_diff'
#  'HT_Hforme' 'HT_Aforme' 'HT_Hdef' 'HT_forme_diff' 'HRepos'
#  'FT_Hprecision' 'FT_Aprecision' 'FT_Hprec_weight' 'FT_Aprec_weight'
#  'FT_prec_diff' 'FT_Elo_H' 'FT_Elo_A' 'FT_Elo_dif' 'HT_Elo_H' 'HT_Elo_A'
#  'HT_Elo_dif' 'FT_avgY_diff' 'FT_avgR_diff' 'FT_forme_ratio'
#  'HT_forme_ratio' 'HT_def_ratio' 'FT_prec_ratio' 'FT_prec_weight_ratio'
#  'FT_Elo_ratio' 'HT_Elo_ratio' 'FT_avgY_ratio' 'FT_avgR_ratio' 'Rank_diff']

# %% PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"Nombre de composantes PCA (95% variance) : {X_pca.shape[1]}")
# Nombre de composantes PCA (95% variance) : 23

# %% ANOVA
selector_anova = SelectKBest(score_func=f_classif, k=rfecv.n_features_)
selector_anova.fit(X_scaled, y)

features_anova = pd.Series(features)[selector_anova.get_support()]
print(f"Features ANOVA : {features_anova.values}")
# Features ANOVA : ['FT_Hforme' 'FT_Aforme' 'FT_Hatt' 'FT_Aatt' 'FT_Hdef' 'FT_Adef'
#  'FT_forme_diff' 'FT_att_diff' 'FT_def_diff' 'HT_Hforme' 'HT_forme_diff'
#  'HT_def_diff' 'FT_prec_diff' 'FT_prec_weight_diff' 'FT_Elo_H' 'FT_Elo_A'
#  'FT_Elo_dif' 'HT_Elo_H' 'HT_Elo_A' 'HT_Elo_dif' 'FT_forme_ratio'
#  'FT_att_ratio' 'FT_def_ratio' 'HT_forme_ratio' 'HT_def_ratio'
#  'FT_prec_ratio' 'FT_prec_weight_ratio' 'FT_Elo_ratio' 'HT_Elo_ratio'
#  'H_WinStreak' 'WinStreak_diff' 'H_Rank' 'A_Rank' 'Rank_diff']

# %% Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=rfecv.n_features_)
selector_mi.fit(X_scaled, y)

features_mi = pd.Series(features)[selector_mi.get_support()]
print(f"Features Mutual Info : {features_mi.values}")
# Features Mutual Info : ['FT_Hforme' 'FT_Aforme' 'FT_Hatt' 'FT_Adef' 'FT_forme_diff' 'FT_att_diff'
#  'FT_def_diff' 'HT_Hforme' 'HT_forme_diff' 'FT_Hprecision' 'FT_Aprecision'
#  'FT_Hprec_weight' 'FT_Aprec_weight' 'FT_prec_diff' 'FT_prec_weight_diff'
#  'FT_Elo_H' 'FT_Elo_A' 'FT_Elo_dif' 'HT_Elo_H' 'HT_Elo_A' 'HT_Elo_dif'
#  'FT_avgF_diff' 'FT_forme_ratio' 'FT_att_ratio' 'FT_def_ratio'
#  'HT_forme_ratio' 'HT_def_ratio' 'FT_prec_ratio' 'FT_Elo_ratio'
#  'HT_Elo_ratio' 'A_WinStreak' 'WinStreak_diff' 'A_Rank' 'Rank_diff']

# %% Comparaison
features_communes = set(features_rfecv) & set(features_anova) & set(features_mi)
print(f"\n--- Comparaison ---")
print(f"RFECV       ({len(features_rfecv)}) : {sorted(features_rfecv.values)}")
print(f"ANOVA       ({len(features_anova)}) : {sorted(features_anova.values)}")
print(f"Mutual Info ({len(features_mi)})  : {sorted(features_mi.values)}")
print(f"Communes 3  ({len(features_communes)}) : {sorted(features_communes)}")

# --- Comparaison ---
# RFECV       (34): ['FT_Aatt', 'FT_Adef', 'FT_Aforme', 'FT_Aprec_weight', 'FT_Aprecision', 'FT_Elo_A', 'FT_Elo_H',
#                    'FT_Elo_dif', 'FT_Elo_ratio', 'FT_Hdef', 'FT_Hforme', 'FT_Hprec_weight', 'FT_Hprecision',
#                    'FT_avgR_diff', 'FT_avgR_ratio', 'FT_avgY_diff', 'FT_avgY_ratio', 'FT_forme_diff', 'FT_forme_ratio',
#                    'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_ratio', 'HRepos', 'HT_Aforme', 'HT_Elo_A',
#                    'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Hdef', 'HT_Hforme', 'HT_def_ratio', 'HT_forme_diff',
#                    'HT_forme_ratio', 'Rank_diff']
#
# ANOVA(34): ['A_Rank', 'FT_Aatt', 'FT_Adef', 'FT_Aforme', 'FT_Elo_A', 'FT_Elo_H', 'FT_Elo_dif', 'FT_Elo_ratio',
#             'FT_Hatt', 'FT_Hdef', 'FT_Hforme', 'FT_att_diff', 'FT_att_ratio', 'FT_def_diff', 'FT_def_ratio',
#             'FT_forme_diff', 'FT_forme_ratio', 'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_diff',
#             'FT_prec_weight_ratio', 'HT_Elo_A', 'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Hforme', 'HT_def_diff',
#             'HT_def_ratio', 'HT_forme_diff', 'HT_forme_ratio', 'H_Rank', 'H_WinStreak', 'Rank_diff', 'WinStreak_diff']
# Mutual
# Info(34): ['A_Rank', 'A_WinStreak', 'FT_Adef', 'FT_Aforme', 'FT_Aprec_weight', 'FT_Aprecision', 'FT_Elo_A', 'FT_Elo_H',
#            'FT_Elo_dif', 'FT_Elo_ratio', 'FT_Hatt', 'FT_Hforme', 'FT_Hprec_weight', 'FT_Hprecision', 'FT_att_diff',
#            'FT_att_ratio', 'FT_avgF_diff', 'FT_def_diff', 'FT_def_ratio', 'FT_forme_diff', 'FT_forme_ratio',
#            'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_diff', 'HT_Elo_A', 'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio',
#            'HT_Hforme', 'HT_def_ratio', 'HT_forme_diff', 'HT_forme_ratio', 'Rank_diff', 'WinStreak_diff']
# Communes
# 3(20): ['FT_Adef', 'FT_Aforme', 'FT_Elo_A', 'FT_Elo_H', 'FT_Elo_dif', 'FT_Elo_ratio', 'FT_Hforme', 'FT_forme_diff',
#         'FT_forme_ratio', 'FT_prec_diff', 'FT_prec_ratio', 'HT_Elo_A', 'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio',
#         'HT_Hforme', 'HT_def_ratio', 'HT_forme_diff', 'HT_forme_ratio', 'Rank_diff']


# =============================================================================
# %% MLP + Boucle Backward ECD (Early Cell Damage)
# =============================================================================
from sklearn.metrics import log_loss

X_train_scaled_full = X_scaled[:split]
y_train_full = y.iloc[:split]

X_test_scaled = X_scaled[split:]
y_test = y.iloc[split:]

# 2. Création d'un sous-set de Validation pour évaluer la boucle backward
# On garde 80% pour l'entraînement interne, 20% pour l'évaluation de la sélection
val_split = int(len(X_train_scaled_full) * 0.8)
X_tr = X_train_scaled_full[:val_split]
y_tr = y_train_full.iloc[:val_split]
X_val = X_train_scaled_full[val_split:]
y_val = y_train_full.iloc[val_split:]


def get_ecd_saliency(nn, X, y):
    """
        Calcule la matrice de saillance (saliency) des variables d'entrée d'un MLP
        via l'approche ECD (Early Cell Damage / Early Brain Damage).

        Cette méthode est une extension au second ordre d'Optimal Cell Damage (OCD).
        Contrairement à OCD/OBD qui supposent un gradient nul (convergence parfaite),
        ECD prend en compte le gradient actuel, ce qui permet d'estimer la pertinence
        des caractéristiques (features) même si le modèle n'a pas atteint un minimum local.

        Le calcul estime l'impact sur la fonction de coût (Log-Loss) de la mise à zéro
        simultanée des poids connectés à une entrée en utilisant un développement de
        Taylor au second ordre.

        Parameters
        ----------
        nn : sklearn.neural_network.MLPClassifier
            Le modèle de réseau de neurones (Perceptron Multicouche) entraîné ou en cours
            d'entraînement. Doit posséder l'attribut `coefs_`.
        X : array-like of shape (n_samples, n_features)
            La matrice des données d'entrée (caractéristiques).
        y : array-like of shape (n_samples,)
            Les étiquettes de classes réelles associées aux données d'entrée.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Un vecteur contenant le score de saillance (saliency) pour chaque variable.
            Une valeur plus élevée indique une variable plus importante pour les prédictions
            du réseau.

        Notes
        -----
        L'approximation des dérivées partielles premières (gradient $g_0$) et secondes
        (Hessien diagonal $h_0$) est réalisée par la méthode des différences finies centrées
        autour des poids de la première couche cachée ($W_0$).

        La formule de saillance pour une connexion $W_{i,j}$ est définie par :
        $$S_{i,j} = \frac{1}{2} h_0 W_0^2 - g_0 W_0 + \frac{1}{2} \frac{g_0^2}{h_0}$$

        La saillance globale de la variable d'entrée $i$ correspond à la somme des saillances
        de toutes ses connexions sortantes (*fan-out*).

        References
        ----------
        .. [1] Leray, P., & Gallinari, P. (1998). "Feature Selection with Neural Networks".
               Research Report LIP6. (Section 5.3.2 - Optimal Cell Damage & Équations ECD).
        .. [2] Hermann, G. (1997). "Early Brain Damage". Neural Information Processing Systems.
        """
    W0 = nn.coefs_[0]  # Poids de la première couche (n_features, n_hidden)
    g0 = np.zeros_like(W0)
    h0 = np.zeros_like(W0)
    eps = 1e-4  # Perturbation pour l'approximation numérique

    # Perte de base
    proba_base = nn.predict_proba(X)
    base_loss = log_loss(y, proba_base)

    # Calcul numérique des gradients (dérivée première) et hessiens (dérivée seconde)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            orig_val = W0[i, j]

            # Perturbation +eps
            nn.coefs_[0][i, j] = orig_val + eps
            loss_plus = log_loss(y, nn.predict_proba(X))

            # Perturbation -eps
            nn.coefs_[0][i, j] = orig_val - eps
            loss_minus = log_loss(y, nn.predict_proba(X))

            # Remise à l'état initial
            nn.coefs_[0][i, j] = orig_val

            # différences finies centrées des dérivées première et seconde
            g0[i, j] = (loss_plus - loss_minus) / (2 * eps)
            h0[i, j] = (loss_plus - 2 * base_loss + loss_minus) / (eps ** 2)

    h0_safe = np.where(np.abs(h0) < 1e-8, 1e-8, h0)

    term_obd = 0.5 * h0 * (W0 ** 2)
    term_gradient = -g0 * W0
    term_correction = 0.5 * (g0 ** 2) / h0_safe

    saliency_matrix = term_obd + term_gradient + term_correction

    # La saliance de l'entrée est la somme des saliances de ses connexions sortantes (fan-out)
    return np.sum(saliency_matrix, axis=1)


def train_nn_early_stopping(X_train, y_train):
    """Entraîne un MLP avec arrêt précoce (respect du postulat ECD)"""
    nn = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=1000,
        early_stopping=True,  # L'arrêt précoce justifie la méthode ECD vs OCD
        validation_fraction=0.1,  # Split interne pour l'early stopping
        n_iter_no_change=10,
        random_state=42
    )
    nn.fit(X_train, y_train)
    return nn


# --- INITIALISATION DE LA BOUCLE BACKWARD ---
active_features_idx = list(range(len(features)))
history = []

print("\n" + "=" * 60)
print("DÉBUT DE LA SÉLECTION BACKWARD ECD")
print("=" * 60)

# Étape Initiale : Entraînement sur toutes les features
nn_current = train_nn_early_stopping(X_tr[:, active_features_idx], y_tr)
y_pred_val = nn_current.predict(X_val[:, active_features_idx])
f1_current = f1_score(y_val, y_pred_val, average='macro')

history.append({
    'n_features': len(active_features_idx),
    'f1_val': f1_current,
    'features_idx': list(active_features_idx)
})

print(f"[Init] {len(active_features_idx)} features | F1-Val: {f1_current:.4f}")

# Boucle d'élimination
best_f1_global = f1_current

while len(active_features_idx) > 1:
    X_tr_active = X_tr[:, active_features_idx]

    # 1. Calcul de la Saliance ECD sur le Train set
    saliences = get_ecd_saliency(nn_current, X_tr_active, y_tr)

    # 2. Identification de la pire variable
    worst_local_idx = int(np.argmin(saliences))
    worst_global_idx = active_features_idx[worst_local_idx]
    removed_feature_name = features[worst_global_idx]

    # 3. Élimination
    active_features_idx.pop(worst_local_idx)

    # 4. Réentraînement avec les variables restantes
    nn_current = train_nn_early_stopping(X_tr[:, active_features_idx], y_tr)

    # 5. Évaluation
    y_pred_val = nn_current.predict(X_val[:, active_features_idx])
    f1_current = f1_score(y_val, y_pred_val, average='macro')

    history.append({
        'n_features': len(active_features_idx),
        'f1_val': f1_current,
        'features_idx': list(active_features_idx)
    })

    print(f"Suppression de: '{removed_feature_name}' | Reste: {len(active_features_idx)} | F1-Val: {f1_current:.4f}")

    # Condition d'arrêt dynamique : Si le modèle s'effondre totalement
    if f1_current > best_f1_global:
        best_f1_global = f1_current
    elif f1_current < (best_f1_global * 0.85):
        print("Chute drastique des performances.")
        break

# --- SÉLECTION DU MODÈLE FINAL (TEST DE FISHER / TOLÉRANCE) ---
history_df = pd.DataFrame(history)
best_f1 = history_df["f1_val"].max()
tol = 0.015  # Marge de tolérance (approximation du test de Fisher)

# On garde les modèles dont la performance est statistiquement "proche" de la meilleure
candidates = history_df[history_df["f1_val"] >= (best_f1 - tol)]

# Principe de parcimonie : on prend le modèle avec le moins de features parmi ces candidats
best_row = candidates.loc[candidates["n_features"].idxmin()]

best_n_features = int(best_row["n_features"])
best_f1_score = best_row["f1_val"]
best_feat_idx = best_row["features_idx"]
features_ecd = [features[i] for i in best_feat_idx]

print("\n" + "=" * 60)
print("RÉSULTAT SÉLECTION ECD")
print("=" * 60)
print(f"Meilleur F1 observé  : {best_f1:.4f}")
print(f"Modèle optimal retenu: {best_n_features} features (F1: {best_f1_score:.4f}, tolérance {tol})")
print(f"Features ECD         : {sorted(features_ecd)}")
# Features ECD: ['FT_Aprec_weight', 'FT_avgY_diff', 'HT_Elo_A', 'HT_Elo_dif', 'HT_Hdef', 'WinStreak_diff']


# --- TEST SUR LE VRAI SET DE TEST INÉDIT ---
# On réentraîne un modèle final propre sur l'intégralité du Train_scaled avec les features sélectionnées
print("\nÉvaluation sur le set de TEST (Matchs non vus)...")
nn_final = train_nn_early_stopping(X_train_scaled_full[:, best_feat_idx], y_train_full)
y_pred_test = nn_final.predict(X_test_scaled[:, best_feat_idx])

print("\n--- Classification Report (TEST SET) ---")
print(classification_report(y_test, y_pred_test))

# Mise à jour des communes avec l'ECD
features_communes_4 = set(features_communes) & set(features_ecd)
print("\n---  Communes aux 4 méthodes ---")
print(f"Total ({len(features_communes_4)}) : {sorted(features_communes_4)}")
