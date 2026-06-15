import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

# 1. Chargement des données
df = pd.read_csv('../data/csv/foot_process.csv', low_memory=False)

# Configuration des bookmakers et de leurs colonnes respectives
books_config = {
    'B365': {'H': 'B365H', 'D': 'B365D', 'A': 'B365A'},
    'Bwin': {'H': 'BWH', 'D': 'BWD', 'A': 'BWA'},
    'Pinnacle': {'H': 'PSH', 'D': 'PSD', 'A': 'PSA'},
    'Pinnacle_Closing': {'H': 'PSCH', 'D': 'PSCD', 'A': 'PSCA'},
    'BetVictor': {'H': 'VCH', 'D': 'VCD', 'A': 'VCA'}
}

# Extraction de toutes les colonnes de cotes nécessaires
all_bet_cols = [col for b in books_config.values() for col in b.values()]

# Nettoyage et copie des colonnes indispensables
df_clean = df[all_bet_cols + ['FTR', 'Season', 'League']].copy()

for col in all_bet_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Suppression des valeurs manquantes ou aberrantes
df_clean = df_clean[(df_clean[all_bet_cols] > 1).all(axis=1)]
df_clean = df_clean.dropna(subset=['Season', 'League', 'FTR'])

print(f"Nombre de matchs après nettoyage : {len(df_clean)}")
print(f"Saisons détectées : {df_clean['Season'].unique()}")
print(f"Ligues détectées : {df_clean['League'].unique()}")


# 2. Fonction de Perte : Log-Loss Binaire (Home vs Rest)
def calculate_home_vs_rest_log_loss(row, book_cols):
    # Extraction des probabilités brutes (inverses des cotes)
    p_h_raw = 1 / row[book_cols['H']]
    p_d_raw = 1 / row[book_cols['D']]
    p_a_raw = 1 / row[book_cols['A']]

    # Normalisation pour retirer la marge commerciale du bookmaker
    total_raw = p_h_raw + p_d_raw + p_a_raw
    p_h = p_h_raw / total_raw

    # Sécurité indispensable pour éviter les erreurs log(0)
    p_h = np.clip(p_h, 1e-15, 1 - 1e-15)

    # Encodage binaire : 1 si Victoire Domicile (H), 0 sinon (D ou A)
    y_true = 1 if row['FTR'] == 'H' else 0

    # Calcul de la perte binaire
    if y_true == 1:
        return -np.log(p_h)
    else:
        return -np.log(1 - p_h)


# 3. Calcul de la Log-Loss match par match
match_losses = pd.DataFrame(index=df_clean.index)
match_losses['Season'] = df_clean['Season']
match_losses['League'] = df_clean['League']

for name, cols in books_config.items():
    match_losses[name] = df_clean.apply(lambda r: calculate_home_vs_rest_log_loss(r, cols), axis=1)

# 4. Aggrégation par couple (Saison, Ligue) pour éviter le MultiIndex et le bug de scikit-posthocs
scores_df = match_losses.groupby(['Season', 'League']).mean()

# Aplatissement du MultiIndex en un Index simple (ex: "2324_Ligue 1")
scores_df.index = [f"{s}_{l}" for s, l in scores_df.index]
scores_df.index.name = None

print("\n--- MATRICE DES SCORES PAR SAISON / LIGUE (Utilisée pour Friedman) ---")
print(scores_df.head(15).round(4))
print(f"... ({len(scores_df)} blocs d'observations au total) ...")
print("-" * 65)

# 5. Test statistique global de Friedman
stat, p_value = stats.friedmanchisquare(*[scores_df[c] for c in scores_df.columns])
print(f"Statistique de Friedman : {stat:.3f}, P-value : {p_value:.5f}")

# 6. Analyse Post-Hoc et Graphique de Différence Critique (CD Diagram)
if p_value < 0.05:
    ranks = scores_df.rank(axis=1).mean()
    print("\nRanking moyen des bookmakers (Plus bas = Meilleur) :")
    print(ranks.sort_values().round(3))

    # Calcul de la matrice de p-values avec le test post-hoc de Nemenyi
    p_matrix = sp.posthoc_nemenyi_friedman(scores_df)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Appel unique de la fonction de tracé
    result = sp.critical_difference_diagram(ranks, p_matrix, ax=ax)

    # Customisation visuelle des barres de non-significativité (crossbars)
    crossbars = result["crossbars"]
    for group in crossbars:
        for line in group:
            x = line.get_xdata()
            y = line.get_ydata()
            x_min, x_max = np.min(x), np.max(x)
            y_val = y[0]
            ax.plot([x_min, x_max], [y_val, y_val], marker='o', markersize=6,
                    color="red", linestyle='None', zorder=10)

    plt.title(
        f"CD Diagram — Comparaison des Bookmakers (Métrique : Log-Loss, H vs Rest)",
        fontsize=12, pad=15)
    plt.tight_layout()
    plt.show()
else:
    print("\n[Pas de différence statistiquement significative décelée (p >= 0.05)]")
