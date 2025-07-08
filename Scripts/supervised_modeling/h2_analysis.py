# h2_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# === CONFIG ===
csv_path = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/PCA_run/analysis/supervised__pca_summary_by_seed.csv"
RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/PCA_run"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
# === LOAD DATA ===
df = pd.read_csv(csv_path)
df['combo'] = df['model'] + " + " + df['cluster_condition']
metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]

anova_results = []

# === ANOVA ON EACH METRIC ===
for metric in metrics:
    model = ols(f"{metric} ~ C(combo)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_val = anova_table["PR(>F)"][0]
    anova_results.append({
        "metric": metric,
        "p_value": p_val
    })

anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(os.path.join(OUTPUT_DIR,"h2_anova_results.csv"), index=False)
anova_df.to_latex(os.path.join(OUTPUT_DIR,"h2_anova_results.tex"), index=False)

# === POST-HOC TEST: Tukey HSD on F1 ===
tukey = pairwise_tukeyhsd(endog=df["f1"], groups=df["combo"], alpha=0.05)
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
tukey_df.to_csv(os.path.join(OUTPUT_DIR,"h2_tukey_f1.csv"), index=False)
tukey_df.to_latex(os.path.join(OUTPUT_DIR,"h2_tukey_f1.tex"), index=False)

# === PLOT F1 DISTRIBUTIONS ===
plt.style.use("/content/drive/MyDrive/NCU/Dissertation/apa.mplstyle")
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x="combo", y="f1", palette="cividis")
plt.xticks(rotation=90)
#plt.title("F1 Score by Model + Cluster Configuration")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"h2_f1_boxplot.png"), dpi=300)
plt.close()

print(" H2 ANOVA + Tukey analysis complete.")
