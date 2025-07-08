import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
csv_path = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/PCA_run/all_features/analysis/supervised__pca_summary_by_seed.csv"
RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/PCA_run/all_features"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
df = pd.read_csv(csv_path)

# Define metrics to evaluate
metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]

# Perform two-way ANOVA with interaction for each metric
anova_results = {}

for metric in metrics:
    formula = f"{metric} ~ C(model) * C(cluster_condition)"
    model = ols(formula, data=df).fit()
    
    # ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[metric] = anova_table
    print(f"\nANOVA Results for {metric}:\n", anova_table)

    # Save ANOVA table
    anova_out_path = os.path.join(OUTPUT_DIR, f"anova_results_{metric}.csv")
    anova_table.to_csv(anova_out_path)

    # Save residuals and fitted values
    df_with_resid = df.copy()
    df_with_resid[f"{metric}_residual"] = model.resid
    df_with_resid[f"{metric}_fitted"] = model.fittedvalues
    resid_out_path = os.path.join(OUTPUT_DIR, f"{metric}_with_residuals.csv")
    df_with_resid.to_csv(resid_out_path, index=False)


# Optional: Save results to CSVs
for metric, table in anova_results.items():
    out_path = os.path.join(OUTPUT_DIR, f"anova_results_{metric}.csv")
    table.to_csv(out_path)

