"""
=============================================================================
VISUALISATION EXPLORATOIRE DES DONNÉES (EDA)
=============================================================================
Script d'exploration visuelle des données de football via réduction de
dimensionnalité et analyses multivariées.

Méthodes de réduction
---------------------
- UMAP (Uniform Manifold Approximation and Projection) : préserve la structure
  locale et globale en 2D et 3D. Très efficace pour découvrir les clusters.

- PCA (Principal Component Analysis) : projection linéaire optimalisant la
  variance expliquée. Interprétabilité plus claire que UMAP.

Analyses
--------
1. UMAP 2D / 3D : Visualisation non-linéaire des séparations 1X2.
2. PCA 2D / 3D : Analyse linéaire + variance par composante.
3. Variance cumulée : Nombre de composantes PCA nécessaires pour 95% variance.
4. Biplot PCA : Projection des observations + contributions des variables.
5. Boxplot : Distributions univariées par classe cible (Hvs).

Target
------
FTR : 0=Draw, 1=Home, 2=Away (multinomial)
Hvs : 0=Not Home Win, 1=Home Win (binaire)

Palette de couleurs : seaborn.color_palette()
  - Indice 0 (bleu) : Draw / Not Home
  - Indice 1 (orange) : Home Win
  - Indice 2 (vert) : Away
"""

import sys
import os

sys.path.append(os.path.abspath('code'))
import config as conf
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib

matplotlib.use('qtagg')
print("Import")

# %%=============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
# Récupération des données nettoyées et sélection des features via config.

df = pd.read_csv(conf.path_clean_cell)
feat = conf.ECD  # Features sélectionnées (config.py)
X = df[feat].values  # Matrice d'observations (n_samples, n_features)
y = df['FTR']  # Cible multinominale (0, 1, 2)

# Normalisation : centrage et réduction à variance unitaire
# Essentielle pour UMAP et PCA
scaled = StandardScaler().fit_transform(X)

print(f" Données chargées : {X.shape[0]} matchs, {X.shape[1]} features")

# %%=============================================================================
# VISUALISATION 1 : UMAP 2D
# =============================================================================
# UMAP : réduction non-linéaire préservant structures locales et globales.
# Paramètres par défaut :
#   - n_neighbors=15 : rayon du voisinage local
#   - min_dist=0.1 : distance minimale entre points projetés
#   - metric='euclidean' : distance source

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(scaled)

# Palette de couleurs et mapping labels
palette = sns.color_palette()
label_map = {0: "Draw", 1: "Home", 2: "Away"}

# Projection 2D colorée par classe
plt.figure(figsize=(10, 8))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[palette[int(x)] for x in df.FTR],
    alpha=0.6,
    s=10,
    edgecolors='none')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
plt.legend(handles=legend_elements, fontsize=10)

plt.gca().set_aspect('equal', 'datalim')
plt.xlabel('UMAP 1', fontsize=10)
plt.ylabel('UMAP 2', fontsize=10)
plt.title('UMAP 2D Projection (séparation 1X2)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(" UMAP 2D ")

# %%=============================================================================
# VISUALISATION 2 : UMAP 3D
# =============================================================================
# UMAP en 3 dimensions pour explorer la séparation spatiale des classes.
# Permet de pivoter/zoomer interactivement (avec QT5).

reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
embedding_3d = reducer_3d.fit_transform(scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = [palette[int(x)] for x in df.FTR]
ax.scatter(
    embedding_3d[:, 0],
    embedding_3d[:, 1],
    embedding_3d[:, 2],
    c=colors,
    s=10,
    alpha=0.6,
    edgecolors='none')

legend_elements_3d = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
ax.legend(handles=legend_elements_3d)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('UMAP 3D Projection (interactif)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Plot 3D affiché")

# %%=============================================================================
# VISUALISATION 3 : PCA 2D
# =============================================================================
# PCA : réduction linéaire via diagonalisation de la matrice de covariance.
# Dimension 1 & 2 capturent le plus de variance possible dans le sous-espace.

pca = PCA(n_components=2, random_state=42)
embedding_pca = pca.fit_transform(scaled)

plt.figure(figsize=(10, 8))
plt.scatter(
    embedding_pca[:, 0],
    embedding_pca[:, 1],
    c=[palette[int(x)] for x in df.FTR],
    alpha=0.6,
    s=10,
    edgecolors='none')

legend_elements_pca = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
plt.legend(handles=legend_elements_pca, fontsize=10)

var_pca1 = pca.explained_variance_ratio_[0] * 100
var_pca2 = pca.explained_variance_ratio_[1] * 100

plt.xlabel(f'PC1 ({var_pca1:.1f}%)', fontsize=10)
plt.ylabel(f'PC2 ({var_pca2:.1f}%)', fontsize=10)
plt.title('PCA 2D Projection', fontsize=14, fontweight='bold')
plt.gca().set_aspect('equal', 'datalim')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f" ACP 2D affiché (variance cumulée : {(var_pca1 + var_pca2):.1f}%)")

# %% ACP 3D
pca_3d = PCA(n_components=3, random_state=42)
embedding_pca_3d = pca_3d.fit_transform(scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    embedding_pca_3d[:, 0],
    embedding_pca_3d[:, 1],
    embedding_pca_3d[:, 2],
    c=colors,
    s=10,
    alpha=0.6
)

legend_elements_pca3d = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
ax.legend(handles=legend_elements_pca3d)

ax.set_title('ACP Projection 3D', fontsize=20)
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.1f}%)')
plt.show()
print("Plot ACP 3D affiché")

# %% Variance expliquée
pca_full = PCA(random_state=42)
pca_full.fit(scaled)

cumvar = pca_full.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', markersize=3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('ACP - Variance expliquée cumulée')
plt.legend()
plt.grid(True)
plt.show()
print("Plot variance expliquée affiché")

# %% Biplot ACP
pca_biplot = PCA(n_components=2, random_state=42)
embedding_biplot = pca_biplot.fit_transform(scaled)

fig, ax = plt.subplots(figsize=(14, 10))

scores = embedding_biplot / np.abs(embedding_biplot).max()

ax.scatter(
    scores[:, 0],
    scores[:, 1],
    c=[palette[x] for x in df.FTR],
    alpha=0.4,
    s=10
)

components = pca_biplot.components_.T
loadings = components / np.abs(components).max()

for i, feature in enumerate(feat):
    ax.arrow(
        0, 0,
        loadings[i, 0],
        loadings[i, 1],
        head_width=0.02,
        head_length=0.02,
        fc='black',
        ec='black',
        alpha=0.8
    )
    offset_x = 0.05 if loadings[i, 0] >= 0 else -0.05
    offset_y = 0.05 if loadings[i, 1] >= 0 else -0.05
    ax.text(
        loadings[i, 0] + offset_x,
        loadings[i, 1] + offset_y,
        feature,
        fontsize=7,
        ha='center',
        va='center',
        color='darkred'
    )

legend_elements_bi = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
ax.legend(handles=legend_elements_bi)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel(f'PC1 ({pca_biplot.explained_variance_ratio_[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_biplot.explained_variance_ratio_[1] * 100:.1f}%)')
ax.set_title('Biplot ACP', fontsize=24)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.show()

# %% BOXPLOT

t = 'Hvs'
plt.figure(figsize=(20, 16))
for i, col in enumerate(feat):
    if len(feat)==20:
        plt.subplot(4, 5, i + 1)
    elif len(feat)==6:
        plt.subplot(2, 3, i + 1)
    sns.boxplot(x=t, y=col, data=df, showfliers=False)
    plt.title(f"{col}", fontsize=12)
    if len(feat) >=15:
        plt.xlabel(None)
    elif len(feat) ==6:
        if t == 'FTR':
            plt.xlabel("1=Dom, 0=Nul, 2=Ext")
        else:
            plt.xlabel("Résultat (0=Perdu/Nul, 1=Gagné)")
plt.suptitle(f"target : {t}", y=0.999)
plt.tight_layout()
plt.show()

# %%