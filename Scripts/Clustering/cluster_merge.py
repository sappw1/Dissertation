import pandas as pd
import numpy as np
import os

# === Paths ===
DATA_DIR = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering"
CLUSTER_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/labels"

# === Load Base Feature Matrix ===
X_PCA_npy = np.load(os.path.join(DATA_DIR, "X_all_pca_3.npy"))
X_base = pd.DataFrame(X_PCA_npy)
#X_base = pd.read_pickle(os.path.join(DATA_DIR, "X_all_scaled.pkl"))
index_df = pd.read_csv(os.path.join(DATA_DIR, "index_all_scaled.csv"), index_col=0)
X_base.index = index_df.index  # Ensure alignment

# === Load Cluster Features ===
kmeans_labels = np.load(os.path.join(CLUSTER_DIR, "kmeans_full_n2_labels.npy"))
hier_labels = np.load(os.path.join(CLUSTER_DIR, "hier_full_n2_labels.npy"))
# Build the binary noise mask from noise indices
noise_indices_path = os.path.join(CLUSTER_DIR, "dbscan_full_e0.7_m10_noise_indices.txt")
with open(noise_indices_path) as f:
    noise_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

dbscan_noise_mask = np.zeros(X_base.shape[0], dtype=int)
dbscan_noise_mask[noise_indices] = 1


# === Inject Columns ===
X_base["kmeans_full_n2"] = kmeans_labels
X_base["hier_full_n2"] = hier_labels
X_base["dbscan_full_e07_m10_noise"] = dbscan_noise_mask

# === Save to Augmented File ===
X_base.to_pickle(os.path.join(DATA_DIR, "X_pca_augmented.pkl"))
print(" Saved: X_pca_augmented.pkl with selected cluster features")
