from configs import experiment_configs
from models import get_model
from utils import load_data, prepare_data, save_results
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

def run_experiment(config, log_file):
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Starting: {config}")
    log_file.write(f"[{start_time.strftime('%H:%M:%S')}] Starting: {config}\n")

    # Load data based on config
    X, y = load_data(use_cluster_features=config["use_cluster_features"])
    X_train, X_test, y_train, y_test = prepare_data(X, y, apply_smote=config["apply_smote"])

    # Get model
    if config["model_name"] == "neural_net":
        model = get_model(config["model_name"], config["use_gpu"], input_dim=X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=0)
        y_pred = model.predict(X_test).flatten()
        y_pred_bin = (y_pred > 0.5).astype(int)
        y_proba = y_pred
    else:
        model = get_model(config["model_name"], config["use_gpu"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = y_pred
        y_pred_bin = y_pred

    # Metrics
    report = classification_report(y_test, y_pred_bin, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    model_id = f"{config['model_name']}_{'with' if config['use_cluster_features'] else 'no'}_clusters"
    results = {
        "model_id": model_id,
        "classification_report": report,
        "roc_auc": auc
    }

    save_results(results, config)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"[{end_time.strftime('%H:%M:%S')}] Completed: {model_id} in {duration:.2f} minutes")
    log_file.write(f"[{end_time.strftime('%H:%M:%S')}] Completed: {model_id} in {duration:.2f} minutes\n")

    return {
        "model": config["model_name"],
        "cluster_features": config["use_cluster_features"],
        "smote": config["apply_smote"],
        "roc_auc": auc,
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    }

# Run all experiments and save outputs
if __name__ == "__main__":
    all_results = []
    results_dir = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run_log.txt")

    with open(log_path, "w") as log_file:
        log_file.write("=== Supervised Modeling Run Log ===\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for cfg in experiment_configs:
            result = run_experiment(cfg, log_file)
            all_results.append(result)

        log_file.write(f"\nAll experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir, "supervised_master_results.csv"), index=False)
    with open(os.path.join(results_dir, "supervised_master_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n All experiments complete. Master results and log file saved.")
