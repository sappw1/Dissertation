import json
import os
from datetime import datetime
from configs import experiment_configs  # Your 7 model setups
from train_and_evaluate import train_and_evaluate_model  # Model training function
from utils import build_dataset_with_clusters  # Function to merge cluster features into X

# Load cluster configurations
with open("model_config.json", "r") as f:
    cluster_configs = json.load(f)

# Timestamped directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
results_dir = f"results/experiment_batch_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Loop over each cluster configuration
for cluster_cfg in cluster_configs:
    cluster_name = cluster_cfg["name"]
    cluster_features = cluster_cfg.get("cluster_features", [])

    print(f"\n=== Running Cluster Config: {cluster_name} ===")

    # Load dataset with selected cluster features added
    X, y = build_dataset_with_clusters(cluster_features)

    # Loop over each model
    for model_cfg in experiment_configs:
        model_name = model_cfg["model_name"]
        use_gpu = model_cfg.get("use_gpu", False)
        apply_smote = model_cfg.get("apply_smote", False)

        run_id = f"{model_name}__{cluster_name.replace(' ', '_')}"
        print(f"→ Running: {run_id}")

        # Train and evaluate
        results = train_and_evaluate_model(
            model_name=model_name,
            X=X,
            y=y,
            use_gpu=use_gpu,
            apply_smote=apply_smote
        )

        # Save results
        output = {
            "run_id": run_id,
            "model_config": model_cfg,
            "cluster_config": cluster_cfg,
            "metrics": results
        }

        save_path = os.path.join(results_dir, f"{run_id}.json")
        with open(save_path, "w") as f_out:
            json.dump(output, f_out, indent=4)

        print(f" Results saved to {save_path}")
