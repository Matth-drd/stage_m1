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

# %%
df = pd.read_csv(conf.path_clean_cell)
reducer = umap.UMAP(random_state=42)
feat = conf.ECD
X = df[feat].values
y = df['FTR']
print("Data loading ")

# %% umap 2D
scaled = StandardScaler().fit_transform(X)

embedding = reducer.fit_transform(scaled)
embedding.shape

palette = sns.color_palette()
label_map = {0: "Draw", 1: "Home", 2: "Away"}

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[palette[x] for x in df.FTR])

from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
plt.legend(handles=legend_elements)

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=24)
plt.show()

print("plot umap 2D")

# %% umap 3D
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)

embedding_3d = reducer_3d.fit_transform(scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors =  [palette[int(x)] for x in df.FTR]
scatter = ax.scatter(
    embedding_3d[:, 0],
    embedding_3d[:, 1],
    embedding_3d[:, 2],
    c=colors,
    s=10,
    alpha=0.6
)

legend_elements_pca3d = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
ax.legend(handles=legend_elements_pca3d)

ax.set_title('UMAP Projection 3D (Interactif)', fontsize=20)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

plt.show()

print("Plot 3D affiché")

# %% ACP 2D
pca = PCA(n_components=2, random_state=42)
embedding_pca = pca.fit_transform(scaled)

plt.figure(figsize=(10, 7))
plt.scatter(
    embedding_pca[:, 0],
    embedding_pca[:, 1],
    c=[palette[x] for x in df.FTR],
    alpha=0.6,
    s=10
)

legend_elements_pca = [Patch(facecolor=palette[k], label=v) for k, v in label_map.items()]
plt.legend(handles=legend_elements_pca)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('ACP projection', fontsize=24)
plt.gca().set_aspect('equal', 'datalim')
plt.legend()
plt.show()
print("Plot ACP 2D affiché")

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
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
plt.show()
print("Plot ACP 3D affiché")

# %% Variance expliquée
pca_full = PCA(random_state=42)
pca_full.fit(scaled)

cumvar = pca_full.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumvar)+1), cumvar, marker='o', markersize=3)
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
ax.set_xlabel(f'PC1 ({pca_biplot.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_biplot.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Biplot ACP', fontsize=24)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.show()


# %% BOXPLOT

t='Hvs'
for i, col in enumerate(feat):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=t, y=col, data=df, showfliers=False)
    plt.title(f"{col}", fontsize=12)
    if t == 'FTR':
        plt.xlabel("1=Dom, 0=Nul, 2=Ext")
    else:
        plt.xlabel("Résultat (0=Perdu/Nul, 1=Gagné)")
    plt.ylabel("Valeur")
plt.suptitle(f"target : {t}",y=0.99)
plt.tight_layout()
plt.show()