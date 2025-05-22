from configs import load_experiment_configs
from models import get_model
from utils import load_data, prepare_data, save_results
from cuml.metrics import roc_auc_score
from sklearn.metrics import classification_report 
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import cupy as cp
import sys
#import cudf


mode = "all"
if len(sys.argv) > 1:
    mode = sys.argv[1].lower()

QUICK_TEST = False  # Flip to False for full experiment

remaining_models = {"xgboost"}

SEEDS = [42, 52, 62, 72, 82]  # Repeat experiments for statistical analysis

def run_experiment(config, seed, log_file):
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Starting: {config}")
    log_file.write(f"[{start_time.strftime('%H:%M:%S')}] Starting: {config}\n")

    # Load data based on config
    sample_frac = 0.001 if QUICK_TEST else 1.0
    X, y = load_data(cluster_features=config["cluster_features"], sample_frac=sample_frac)
    X_train, X_test, y_train, y_test = prepare_data(X, y, apply_smote=config["apply_smote"],test_size=0.3, random_state=seed)

    if config["use_gpu"] and config["model_name"] == "naive_bayes":
        print(f"Converting to cupy for naive_bayes")

        X_train = cp.asarray(X_train)
        X_test = cp.asarray(X_test)

        y_train = cp.asarray(y_train)
        y_test = cp.asarray(y_test)

    elif config["use_gpu"] and config["model_name"] in ["svm", "rf","xgboost"]:
        print(f"Converting to CuPy arrays for {config['model_name']} (GPU)")
        X_train = cp.asarray(X_train)
        X_test = cp.asarray(X_test)
        y_train = cp.asarray(y_train)
        y_test = cp.asarray(y_test)

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
    print(f"Made it this far")
    # Metrics
    report = classification_report(cp.asnumpy(y_test), cp.asnumpy(y_pred_bin), output_dict=True)
    print(f"Made it this far2")
    auc = roc_auc_score(y_test, y_proba)
    print(f"Made it this far3")
    model_id = f"{config['model_name']}_{config['cluster_condition']}_seed{seed}"

        # Save per-sample predictions
    preds_df = pd.DataFrame({
        "y_true": cp.asnumpy(y_test),
        "y_pred": cp.asnumpy(y_pred_bin),
        "y_proba": cp.asnumpy(y_proba)
    })
    print(f"Home stretch")
    preds_df.to_csv(os.path.join(results_dir, f"{model_id}_predictions.csv"), index=False)

    # Save full result with metadata
    results = {
        "model_id": model_id,
        "model": config["model_name"],
        "cluster_condition": config["cluster_condition"],
        "apply_smote": config["apply_smote"],
        "seed": seed,
        "roc_auc": auc,
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "classification_report": report  # optionally include full report
    }

    save_results(results, config)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"[{end_time.strftime('%H:%M:%S')}] Completed: {model_id} in {duration:.2f} minutes")
    log_file.write(f"[{end_time.strftime('%H:%M:%S')}] Completed: {model_id} in {duration:.2f} minutes\n")

    return results

# Run all experiments and save outputs
if __name__ == "__main__":
    all_results = []
    results_dir = f"/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/{mode.upper()}"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"run_log{mode}.txt")

    existing_results = set([
        f.split(".json")[0]
        for f in os.listdir(results_dir)
        if f.endswith(".json")
    ])

    with open(log_path, "w") as log_file:
        log_file.write("=== Supervised Modeling Run Log ===\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for cfg in load_experiment_configs(mode=mode):
            if cfg['model_name'] not in remaining_models:
                continue

            model_id = f"{cfg['model_name']}__{cfg['cluster_condition'].replace(' ', '_')}__seed{cfg.get('seed', 0)}"
            if model_id in existing_results:
                print(f"[SKIP] Already completed: {model_id}")
                continue
            for seed in SEEDS:
                try:
                    result = run_experiment(cfg, seed, log_file)
                    all_results.append(result)
                except Exception as e:
                    error_msg = f"[ERROR] {cfg['model_name']} (seed={seed}) failed: {str(e)}\n"
                    print(error_msg)
                    log_file.write(error_msg)
        log_file.write(f"\nAll experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir, "supervised_master_results.csv"), index=False)
    with open(os.path.join(results_dir, "supervised_master_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n All experiments complete. Master results and log file saved.")
