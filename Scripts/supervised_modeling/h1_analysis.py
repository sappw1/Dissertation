# h1_analysis.py

import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# === CONFIG ===
csv_path = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/analysis/supervised_summary_by_seed.csv"
RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Define metrics to test
metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
models = df["model"].unique()
cluster_conditions = df["cluster_condition"].unique()
baseline_df = df[df["cluster_condition"] == "Baseline"]

# === PERFORM WILCOXON TESTS ===
results = []
for metric in metrics:
    for model in models:
        base_vals = baseline_df[baseline_df["model"] == model][metric].values
        for cond in cluster_conditions:
            if cond == "Baseline":
                continue
            comp_vals = df[(df["model"] == model) & (df["cluster_condition"] == cond)][metric].values
            if len(base_vals) == len(comp_vals) and len(base_vals) > 0:
                if np.allclose(base_vals, comp_vals): continue
                try:
                    stat, p = stats.wilcoxon(comp_vals, base_vals)
                    diff = np.mean(comp_vals) - np.mean(base_vals)
                    results.append({
                        "model": model,
                        "metric": metric,
                        "cluster_condition": cond,
                        "mean_diff": diff,
                        "p_value": p
                    })
                except Exception as e:
                    print(f"[WARN] {model} {cond} {metric}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR,"h1_results.csv"), index=False)
results_df.to_latex(os.path.join(OUTPUT_DIR,"h1_results.tex"), index=False)
print(" H1 Wilcoxon analysis complete.")
