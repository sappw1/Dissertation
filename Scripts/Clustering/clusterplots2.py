import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# Load APA style if desired
plt.style.use("/content/drive/MyDrive/NCU/Dissertation/apa.mplstyle")

# === Paths ===
LABELS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/labels/key"
OVERLAY_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/analysis/fraud_overlay/key"
X_PCA_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA/X_all_pca_3.npy"
X_FULL_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA/X_all_no_pca.npy"
Y_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/y_labels.pkl"
OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Figures/Clustering/Overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
X_pca = np.load(X_PCA_PATH)
X_no_pca = np.load(X_FULL_PATH)
y = pd.read_pickle(Y_PATH).values

# === List your 3 algorithms here ===
algorithms = ["kmeans_full_n2", "hier_full_n2", "dbscan_full_e0.7_m10"]

# Initialize figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Track whether noise exists in any plot
has_noise = False

for ax, label_name in zip(axs, algorithms):
    labels_path = os.path.join(LABELS_DIR, f"{label_name}_labels.npy")
    if not os.path.exists(labels_path):
        print(f"[Skip] Missing: {labels_path}")
        continue

    labels = np.load(labels_path)
    X_proj = X_pca[:, :2]

    is_noise = labels == -1
    is_cluster = labels != -1
    if np.any(is_noise):
        has_noise = True

    # Plot clusters
    ax.scatter(
        X_proj[is_cluster, 0], X_proj[is_cluster, 1],
        c=labels[is_cluster], cmap="cividis", s=6, alpha=0.5
    )

    # Plot noise
    if np.any(is_noise):
        ax.scatter(
            X_proj[is_noise, 0], X_proj[is_noise, 1],
            c="gray", alpha=0.3, s=6, marker="v"
        )

    # Plot fraud
    ax.scatter(
        X_proj[y == 1, 0], X_proj[y == 1, 1],
        c="red", s=30, marker="x"
    )

    ax.set_title(label_name.replace("_", " ").title(), fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

# Custom legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Clusters', markerfacecolor='black', markersize=6, alpha=0.5),
    Line2D([0], [0], marker='x', color='red', label='Fraud', linestyle='None', markersize=7)
]

if has_noise:
    legend_elements.insert(1, Line2D([0], [0], marker='v', color='gray', label='Noise', linestyle='None', markersize=6, alpha=0.6))

# Add shared legend below all plots
fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), fontsize=9, bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
fig.subplots_adjust(bottom=0.18)  # Make room for legend
fig.savefig(os.path.join(OUTPUT_DIR, "combined_fraud_overlay.png"), dpi=300, bbox_inches="tight")
plt.close()

print(" Combined figure with clean legend saved.")