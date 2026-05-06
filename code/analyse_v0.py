import sys
import os

sys.path.append(os.path.abspath('code'))
import config as conf
import pandas as pd

from sklearn.feature_selection import RFECV
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(conf.path_clean)

features = conf.features
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['Hvs']  # on fixe la cible sur Hvs suite à de premières analyses
# on constate que la colonne Hvs permet d'obtenir de meilleurs résultats que les autres.

models_rfe = {
    "Logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1),
}

resultats_features = {}

for name, model in tqdm(models_rfe.items()):
    print(f"\n{name}")
    selector = RFECV(model, min_features_to_select=5, step=1, n_jobs=-1)

    # Utilise X_scaled pour Logistic, mais X (original) pour RandomForest
    if name == "Logistic":
        X_input = X_scaled
    else:
        X_input = X

    selector = selector.fit(X_input, y)

    selected_features = X.columns[selector.support_].tolist()
    resultats_features[name] = selected_features

for name, f_list in resultats_features.items():
    print(f"\n{name} : Nombre optimal de features trouvé = {selector.n_features_}")
    print(f_list)

list_lr = resultats_features.get("Logistic", [])
list_rf = resultats_features.get("RandomForest", [])
print(f"lr:{list_lr}")
print(f"rf:{list_rf}")
ft = []
nn = []
for f in list_lr:
    if f in list_rf:
        ft.append(f)
    else:
        nn.append(f)

print("=" * 15)
print("=" * 15)
print(f"features communes :{ft}")
print("=" * 15)
print(nn)

# ft = [
# 'FT_Hprecision', 'FT_Hprec_weight', 'FT_Aprec_weight', 'FT_prec_diff',
# 'FT_prec_weight_diff', 'FT_Elo_H', 'FT_Elo_A', 'FT_Elo_dif', 'HT_Elo_H',
# 'HT_Elo_A', 'HT_Elo_dif', 'HT_Elo_ratio', 'FT_avgY_ratio']

# ft=['FT_Elo_H', 'FT_Elo_A', 'FT_Elo_dif', 'HT_Elo_H',
# 'HT_Elo_dif', 'FT_prec_ratio']
#

print(len(ft))

# LOGISTIC = ['FT_forme_diff', 'FT_Adef', 'FT_Hforme', 'FT_prec_weight_diff', 'FT_prec_weight_ratio', 'FT_avgY_ratio',
#             'FT_Aforme', 'FT_Hprecision', 'FT_AavgF', 'HT_forme_diff', 'FT_forme_ratio', 'FT_Aatt', 'FT_def_diff',
#             'HRepos', 'FT_prec_ratio', 'FT_Hprec_weight', 'FT_Hdef', 'Repos_diff', 'Repos_ratio', 'HT_Hdef',
#             'FT_Aprecision', 'ARepos', 'FT_AavgY']
#
# RF = ['FT_avgF_ratio', 'FT_Elo_ratio', 'FT_att_ratio']


# On récupère le classement du dernier modèle (Random Forest)
ranking_rf = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': selector.ranking_
}).sort_values(by='Ranking', ascending=False)
# Les 10 pires (celles éliminées en premier)
pires_features = ranking_rf.head(10)['Feature'].tolist()
meilleures_features = ft[:10]  # On en prend 10 pour l'affichage
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_comparison(df, feature_list, title, target='Hvs'):
    if not feature_list:
        print(f"Pas de features pour {title}")
        return
    n_features = len(feature_list)
    cols = 5
    rows = (n_features + cols - 1) // cols
    plt.figure(figsize=(20, 4 * rows))
    plt.suptitle(title, fontsize=20, y=.98, fontweight='bold')
    for i, col in enumerate(feature_list):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=target, y=col, data=df, showfliers=False)
        plt.title(f"Impact de {col}", fontsize=12)
        plt.xlabel("Résultat Hvs (0 ou 1)")
        plt.ylabel("Valeur")
    plt.tight_layout()
    plt.show()


plot_feature_comparison(df, meilleures_features, "TOP 10 : Meilleures Features (Consensus)")
plot_feature_comparison(df, pires_features, "TOP 10 : Pires Features (Selon RF Ranking)")
