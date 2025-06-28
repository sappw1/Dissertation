import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load APA style if desired
plt.style.use("/content/drive/MyDrive/NCU/Dissertation/apa.mplstyle")

# === Paths ===
LABELS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/labels"
OVERLAY_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/analysis/fraud_overlay"
X_PCA_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA/X_all_pca_3.npy"
X_FULL_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA/X_all_no_pca.npy"
Y_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/y_labels.pkl"

OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Figures/Clustering/Overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
X_pca = np.load(X_PCA_PATH)
X_no_pca = np.load(X_FULL_PATH)
y = pd.read_pickle(Y_PATH).values

# Find all individual fraud overlay files
overlay_files = [f for f in os.listdir(OVERLAY_DIR) if f.endswith(".csv")]

for overlay_file in overlay_files:
    label_name = overlay_file.replace("_fraud_clusters.csv", "")
    labels_path = os.path.join(LABELS_DIR, f"{label_name}_labels.npy")

    if not os.path.exists(labels_path):
        print(f"[Skip] No label file for {label_name}")
        continue

    labels = np.load(labels_path)

    # Select projection
    if "pca" in label_name:
        X_proj = X_pca[:, :2]
    else:
        X_proj = PCA(n_components=2).fit_transform(X_no_pca)

    fig, ax = plt.subplots(figsize=(6, 5))

    # DBSCAN noise handling
    is_noise = labels == -1
    is_cluster = labels != -1

    # Plot clusters
    ax.scatter(
        X_proj[is_cluster, 0],
        X_proj[is_cluster, 1],
        c=labels[is_cluster],
        cmap="cividis",
        s=6,
        alpha=0.5,
        label="Clusters"
    )

    # Plot DBSCAN noise
    if np.any(is_noise):
        ax.scatter(
            X_proj[is_noise, 0],
            X_proj[is_noise, 1],
            c="gray",
            alpha=0.3,
            s=6,
            marker="v",
            label="Noise"            
        )

    # Overlay fraud
    ax.scatter(
        X_proj[y == 1, 0],
        X_proj[y == 1, 1],
        c="red",
        s=30,
        marker="x",
        label="Fraud"
    )

    ax.set_title(f"{label_name} – Fraud Overlay", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{label_name}_fraud_overlay.png"), dpi=300)
    plt.close()

print(f"✅ Completed all overlays. Saved to: {OUTPUT_DIR}")
