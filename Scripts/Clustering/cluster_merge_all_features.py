import pandas as pd
import numpy as np
import os

# === Paths ===
DATA_DIR = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering"
CLUSTER_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/labels"

# === Load Base Feature Matrix ===
X_base = pd.DataFrame(np.load(os.path.join(DATA_DIR, "X_all_pca_3.npy")))
index_df = pd.read_csv(os.path.join(DATA_DIR, "index_all_scaled.csv"), index_col=0)
X_base.index = index_df.index  # Ensure alignment

# === Load Cluster Labels Dynamically ===
for fname in os.listdir(CLUSTER_DIR):
    fpath = os.path.join(CLUSTER_DIR, fname)
    
    if fname.endswith(".npy"):
        key = fname.replace(".npy", "")
        X_base[key] = np.load(fpath)
    
    elif "noise_indices" in fname and fname.endswith(".txt"):
        # Construct a binary mask from noise indices
        key = fname.replace(".txt", "")
        noise_mask = np.zeros(X_base.shape[0], dtype=int)
        with open(fpath) as f:
            indices = [int(line.strip()) for line in f if line.strip().isdigit()]
            noise_mask[indices] = 1
        X_base[key] = noise_mask

# === Save to Augmented File ===
out_path = os.path.join(DATA_DIR, "X_pca_augmented_all_clusters.pkl")
X_base.to_pickle(out_path)
print(f"Saved: {out_path} with all cluster-derived features")
