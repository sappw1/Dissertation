# analyze_supervised.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === PATHS ===
RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Load all results ===
results = []
for file in os.listdir(RESULTS_DIR):
    if file.endswith(".json") and file != "supervised_master_results.json":
        with open(os.path.join(RESULTS_DIR, file)) as f:
            result = json.load(f)
            results.append(result)

# Convert to DataFrame
df = pd.DataFrame(results)

# Save raw version
df.to_csv(os.path.join(OUTPUT_DIR, "supervised_raw_results.csv"), index=False)

# === 2. Basic per-model summary (seed-level, no aggregation yet) ===
summary_cols = ["model", "cluster_condition", "seed", "roc_auc", "accuracy", "precision", "recall", "f1"]
summary_df = df[summary_cols]

# Save tidy version
summary_df.to_csv(os.path.join(OUTPUT_DIR, "supervised_summary_by_seed.csv"), index=False)

# === 3. Per-model × cluster mean/std metrics (for preview only — not hypothesis testing) ===
metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
agg_df = summary_df.groupby(["model", "cluster_condition"])[metrics].agg(["mean", "std"])
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df = agg_df.reset_index()

agg_df.to_csv(os.path.join(OUTPUT_DIR, "supervised_aggregated_metrics.csv"), index=False)

# === 4. Quick F1 comparison plot (for dev only) ===
plt.figure(figsize=(12, 6))
sns.boxplot(data=summary_df, x="model", y="f1", hue="cluster_condition")
plt.title("F1 Score by Model and Cluster Feature Configuration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_f1_model_cluster.png"), dpi=300)
plt.close()

print(" Supervised results parsed and exported. Ready for statistical testing.")
