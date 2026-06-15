import os
import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('code'))
import config as conf

df = pd.read_csv(conf.path_clean)
features_importante = conf.ft_commune


def plot_boxplot(df, features, target='FTR'):
    num_features = len(features)
    if num_features == 0:
        print("Aucune feature à afficher.")
        return

    n_cols = 5
    n_rows = math.ceil(num_features / n_cols)

    plt.figure(figsize=(22, 4 * n_rows))  # La hauteur s'adapte au nombre de lignes
    plt.suptitle(f"Distribution des Features Clés par rapport à : {target}", fontsize=20, y=0.99)

    for i, col in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)

        order = sorted(df[target].dropna().unique())

        sns.boxplot(x=target, y=col, data=df, showfliers=False, order=order)
        plt.title(f"{col}", fontsize=12)
        if target == 'FTR':
            plt.xlabel("Résultat du Match (H=Dom, D=Nul, A=Ext)")
        else:
            plt.xlabel(f"Classe de {target}")

        plt.ylabel("Valeur")

    plt.tight_layout()
    plt.show()


for t in conf.targets:
    plot_boxplot(df, features_importante, target=t)
