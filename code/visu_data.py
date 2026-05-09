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

X = df[conf.features].values
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

print("plot umap")

# %% umap 3D
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)

embedding_3d = reducer_3d.fit_transform(scaled)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = [sns.color_palette("viridis", 2)[int(x)] for x in df.FTR.fillna(0)]

scatter = ax.scatter(
    embedding_3d[:, 0],
    embedding_3d[:, 1],
    embedding_3d[:, 2],
    c=colors,
    s=10,
    alpha=0.6
)

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

