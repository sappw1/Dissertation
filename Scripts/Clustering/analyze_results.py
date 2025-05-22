import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("/content/drive/MyDrive/NCU/Dissertation/apa.mplstyle")

# === CONFIGURE PATHS ===
RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised"
LABELS_DIR = os.path.join(RESULTS_DIR, "labels")
DATA_DIR = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
FRAUD_OVERLAY_DIR = os.path.join(OUTPUT_DIR, "fraud_overlay")
FIG_OVERLAY_DIR = os.path.join(OUTPUT_DIR, "fig_cluster_overlay")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAUD_OVERLAY_DIR, exist_ok=True)
os.makedirs(FIG_OVERLAY_DIR, exist_ok=True)

# === LOAD BASE DATA ===
y = pd.read_pickle(os.path.join(DATA_DIR, "y_labels.pkl"))
X_no_pca = np.load(os.path.join(DATA_DIR, "X_all_no_pca.npy"))
X_pca = np.load(os.path.join(DATA_DIR, "X_all_pca_3.npy"))
total_frauds = y.sum()

# === LOAD RESULTS ===
results = []
for file in os.listdir(RESULTS_DIR):
    if file.endswith("_results.json"):
        with open(os.path.join(RESULTS_DIR, file)) as f:
            results.append(json.load(f))

df = pd.DataFrame(results)
df["n_clusters"] = df["params"].apply(lambda x: x.get("n_clusters", None))
df["eps"] = df["params"].apply(lambda x: x.get("eps", None))
df["min_samples"] = df["params"].apply(lambda x: x.get("min_samples", None))

# === ANALYZE CLUSTERS + FRAUD ===
for row in df.itertuples():
    label_path = os.path.join(LABELS_DIR, f"{row.label_name}_labels.npy")
    if not os.path.exists(label_path): continue

    labels = np.load(label_path)
    merged = pd.DataFrame({"cluster": labels, "fraud": y})
    cluster_stats = merged.groupby("cluster")["fraud"].agg(
        count="count", fraud_sum="sum", fraud_rate="mean"
    ).reset_index()

    cluster_stats["label_name"] = row.label_name
    cluster_stats["algorithm"] = row.algorithm
    cluster_stats["feature_space"] = row.feature_space
    cluster_stats.to_csv(os.path.join(FRAUD_OVERLAY_DIR, f"{row.label_name}_fraud_clusters.csv"), index=False)

    df.loc[row.Index, "fraud_cluster_max"] = cluster_stats["fraud_rate"].max()
    df.loc[row.Index, "fraud_cluster_avg"] = cluster_stats["fraud_rate"].mean()

    # === DBSCAN NOISE ANALYSIS ===
    if row.algorithm == "dbscan":
        noise_path = os.path.join(LABELS_DIR, f"{row.label_name}_noise_indices.txt")
        if os.path.exists(noise_path):
            with open(noise_path) as f:
                noise_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

            n_noise = len(noise_indices)
            fraud_in_noise = y.iloc[noise_indices].sum()
            fraud_noise_rate = fraud_in_noise / n_noise if n_noise else 0
            fraud_capture_pct = fraud_in_noise / total_frauds if total_frauds else 0
            noise_flag = fraud_noise_rate >= 0.10

            df.loc[row.Index, "n_noise"] = n_noise
            df.loc[row.Index, "fraud_in_noise"] = fraud_in_noise
            df.loc[row.Index, "fraud_noise_rate"] = fraud_noise_rate
            df.loc[row.Index, "fraud_capture_pct"] = fraud_capture_pct
            df.loc[row.Index, "noise_feature_flag"] = noise_flag

            if noise_flag:
                noise_mask = np.zeros_like(y)
                noise_mask[noise_indices] = 1
                np.save(os.path.join(FRAUD_OVERLAY_DIR, f"{row.label_name}_noise_mask.npy"), noise_mask)

# === SAVE SUMMARY TABLES ===
df.to_csv(os.path.join(OUTPUT_DIR, "all_config_summary.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "all_config_summary.tex"), "w") as f:
    f.write(df.to_latex(index=False, float_format="%.3f"))

# === FRAUD OVERLAY PLOTS (TOP CONFIGS) ===
# === Robust Plot Selection ===
df_vis = pd.DataFrame()

# 1. Top silhouette per algorithm × feature space
for algo in ['kmeans', 'hierarchical', 'dbscan']:
    for space in ['pca', 'no_pca']:
        subset = df[(df['algorithm'] == algo) & (df['feature_space'] == space)]
        top = subset[subset['silhouette_score'].notna()].sort_values(by='silhouette_score', ascending=False).head(1)
        df_vis = pd.concat([df_vis, top])

# 2. Add DBSCAN configs where noise was useful
df_vis = pd.concat([df_vis, df[(df["algorithm"] == "dbscan") & (df["noise_feature_flag"] == True)]])

# 3. Add config with highest fraud_cluster_max
df_vis = pd.concat([df_vis, df.nlargest(1, "fraud_cluster_max")])

# Drop duplicates
df_vis = df_vis.drop_duplicates(subset="label_name")

# === Generate Overlay Plots ===
for row in df_vis.itertuples():
    labels_path = os.path.join(LABELS_DIR, f"{row.label_name}_labels.npy")
    if not os.path.exists(labels_path): continue

    labels = np.load(labels_path)
    if row.feature_space == "no_pca":
        proj = PCA(n_components=2).fit_transform(X_no_pca)
    else:
        proj = X_pca[:, :2]

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="cividis", s=8, alpha=0.6)
    plt.scatter(proj[y == 1, 0], proj[y == 1, 1], c='red', s=30, marker='x', label="Fraud")
    plt.title(f"{row.label_name} (Silhouette={row.silhouette_score:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OVERLAY_DIR, f"{row.label_name}_overlay.png"), dpi=300)
    plt.close()

print(" analyze_results.py complete. Tables and overlays saved.")
