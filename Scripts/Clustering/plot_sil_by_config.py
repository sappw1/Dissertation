import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# === Setup ===
plt.style.use("/content/drive/MyDrive/NCU/Dissertation/apa.mplstyle")
sns.set_context("notebook")

# Load and convert 'params' from string to dict
df = pd.read_csv("/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised/analysis/all_config_summary.csv")
df["params"] = df["params"].apply(ast.literal_eval)

# Output path
OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Figures/Clustering/Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hatching and grayscale
colors = ['#444444', '#bbbbbb']
hatches = ['/', '\\']
hue_order = ['no_pca', 'pca']

# === Plot by algorithm ===
algorithms = df["algorithm"].unique()

for algo in algorithms:
    subset = df[df["algorithm"] == algo].copy()

    # Generate compact config labels
    def get_config_label(row):
        p = row["params"]
        if algo in ["kmeans", "hierarchical"]:
            return f"n{p['n_clusters']}"
        elif algo == "dbscan":
            return f"e{p['eps']}_m{p['min_samples']}"
        return "cfg"

    subset["config_label"] = subset.apply(get_config_label, axis=1)


    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=False)

    # === Create side-by-side plots ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

    for i, metric in enumerate(["silhouette_score", "dbi_score"]):
        ax = axs[i]
        sns.barplot(
            data=subset,
            x="config_label",
            y=metric,
            hue="feature_space",
            palette=colors,
            ax=ax,
            width=0.5
        )


        for k, bar in enumerate(ax.patches):
            hatch_idx = k % len(hatches)
            bar.set_hatch(hatches[hatch_idx])
            bar.set_edgecolor("black")

        ax.set_title("Silhouette" if metric == "silhouette_score" else "DBI", fontsize=10)
        ax.set_xlabel("Config", fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend_.remove()
    # Create shared legend from first axis
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=9, title="Feature Space", title_fontsize=10, frameon=False)

    #fig.suptitle(f"{algo.upper()} : Clustering Quality by Config", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUTPUT_DIR, f"fig_clustering_metrics_{algo}.png"), dpi=300)
    plt.close()

print(" Silhouette + DBI plots saved to:", OUTPUT_DIR)
