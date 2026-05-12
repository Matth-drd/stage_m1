"""
Ce code permet de comparer une regression logistique avec toutes les features et
seulement les 20 features sélectionnés.
"""
import sys
import os

sys.path.append(os.path.abspath('code'))
import config as conf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv(conf.path_clean)

features = conf.features #toutes les features
ft_co=conf.ft_commune #20 meilleurs features

split_idx = int(len(df) * 0.8)

for t in conf.targets:
    y = df[t]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #  MODÈLE COMPLET
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(df[features].iloc[:split_idx])
    X_test_full = scaler_full.transform(df[features].iloc[split_idx:])

    rl_full = LogisticRegression(max_iter=1000, random_state=42)
    rl_full.fit(X_train_full, y_train)
    y_pred_full = rl_full.predict(X_test_full)
    report_full = classification_report(y_test, y_pred_full)

    #  MODÈLE RÉDUIT
    scaler_com = StandardScaler()
    X_train_com = scaler_com.fit_transform(df[ft_co].iloc[:split_idx])
    X_test_com = scaler_com.transform(df[ft_co].iloc[split_idx:])

    rl_com = LogisticRegression(max_iter=1000, random_state=42)
    rl_com.fit(X_train_com, y_train)
    y_pred_com = rl_com.predict(X_test_com)
    report_com = classification_report(y_test, y_pred_com)

    print(f"\n" + "=" * 100)
    print(f"COMPARAISON DES MODÈLES POUR LA CIBLE : {t}")
    print(f"{'MODÈLE COMPLET (All Features)':<50} | {'MODÈLE RÉDUIT (Communes)':<50}")
    print("=" * 100)

    lines_full = report_full.split('\n')
    lines_com = report_com.split('\n')

    for l1, l2 in zip(lines_full, lines_com):
        print(f"{l1:<50} | {l2:<50}")



# ====================================================================================================
# COMPARAISON DES MODÈLES POUR LA CIBLE : FTR
# MODÈLE COMPLET (All Features)                         | MODÈLE RÉDUIT (Communes)
# ====================================================================================================
#               precision    recall  f1-score   support |               precision    recall  f1-score   support
#                                                    |
#            0       0.50      0.01      0.01       898 |            0       0.00      0.00      0.00       898
#            1       0.55      0.82      0.65      1579 |            1       0.54      0.82      0.65      1579
#            2       0.50      0.55      0.52      1081 |            2       0.51      0.56      0.53      1081
#                                                    |
#     accuracy                           0.53      3558 |     accuracy                           0.53      3558
#    macro avg       0.52      0.46      0.40      3558 |    macro avg       0.35      0.46      0.39      3558
# weighted avg       0.52      0.53      0.45      3558 | weighted avg       0.40      0.53      0.45      3558
#                                                    |
#
# ====================================================================================================
# COMPARAISON DES MODÈLES POUR LA CIBLE : Hvs
# MODÈLE COMPLET (All Features)                      | MODÈLE RÉDUIT (Communes)
# ====================================================================================================
#               precision    recall  f1-score   support |               precision    recall  f1-score   support
#                                                    |
#            0       0.67      0.74      0.70      1979 |            0       0.68      0.73      0.70      1979
#            1       0.62      0.55      0.58      1579 |            1       0.62      0.57      0.60      1579
#                                                    |
#     accuracy                           0.65      3558 |     accuracy                           0.66      3558
#    macro avg       0.65      0.64      0.64      3558 |    macro avg       0.65      0.65      0.65      3558
# weighted avg       0.65      0.65      0.65      3558 | weighted avg       0.65      0.66      0.66      3558
#                                                    |
#
# ====================================================================================================
# COMPARAISON DES MODÈLES POUR LA CIBLE : Avs
# MODÈLE COMPLET (All Features)                      | MODÈLE RÉDUIT (Communes)
# ====================================================================================================
#               precision    recall  f1-score   support |               precision    recall  f1-score   support
#                                                    |
#            0       0.74      0.91      0.82      2477 |            0       0.75      0.91      0.82      2477
#            1       0.58      0.29      0.38      1081 |            1       0.58      0.29      0.39      1081
#                                                    |
#     accuracy                           0.72      3558 |     accuracy                           0.72      3558
#    macro avg       0.66      0.60      0.60      3558 |    macro avg       0.66      0.60      0.60      3558
# weighted avg       0.69      0.72      0.69      3558 | weighted avg       0.70      0.72      0.69      3558
#                                                    |
#
# ====================================================================================================
# COMPARAISON DES MODÈLES POUR LA CIBLE : Dvs
# MODÈLE COMPLET (All Features)                      | MODÈLE RÉDUIT (Communes)
# ====================================================================================================
#               precision    recall  f1-score   support |               precision    recall  f1-score   support
#                                                    |
#            0       0.75      1.00      0.86      2660 |            0       0.75      1.00      0.86      2660
#            1       0.00      0.00      0.00       898 |            1       0.00      0.00      0.00       898
#                                                    |
#     accuracy                           0.75      3558 |     accuracy                           0.75      3558
#    macro avg       0.37      0.50      0.43      3558 |    macro avg       0.37      0.50      0.43      3558
# weighted avg       0.56      0.75      0.64      3558 | weighted avg       0.56      0.75      0.64      3558
#                                                    |