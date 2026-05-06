import os
import sys

sys.path.append(os.path.abspath('code'))
import config as conf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(conf.path_clean)

features_importante = conf.features_importante


def plot_importance_boxplots(df, features, target='FTR'):
    plt.figure(figsize=(22, 10))
    plt.suptitle(f"Distribution des Features Clés par rapport à {target}", fontsize=20, y=0.98)

    for i, col in enumerate(features):
        plt.subplot(2, 5, i + 1)
        sns.boxplot(x=target, y=col, data=df, showfliers=False)

        plt.title(f"Impact de {col}", fontsize=12)
        plt.xlabel("Résultat (0=Perdu/Nul, 1=Gagné)")
        plt.ylabel("Valeur")

    plt.tight_layout()
    plt.show()


for t in conf.targets:
    plot_importance_boxplots(df, features_importante, target=t)
