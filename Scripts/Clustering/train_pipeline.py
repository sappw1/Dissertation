# train_pipeline.py
import os
import json
import numpy as np
from datetime import datetime
from configs import load_clustering_configs
from clustering_algorithms import run_clustering_algorithm
from metrics import compute_clustering_metrics
from utils import load_feature_data, save_clustering_outputs

QUICK_TEST = False  # Flip to False for full-scale run


RESULTS_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Unsupervised"
LOG_FILE = os.path.join(RESULTS_DIR, "unsupervised_run_log.txt")
LABELS_DIR = os.path.join(RESULTS_DIR, "labels")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

def run_clustering_job(config, log_file):
    start = datetime.now()
    run_id = f"{config['algorithm']}__{config['feature_space']}__{config.get('label_name', 'default')}"
    log_file.write(f"\n[{start.strftime('%H:%M:%S')}] Starting {run_id}\n")
    print(f"[{start.strftime('%H:%M:%S')}] Running: {run_id}")

    # Load data
    X = load_feature_data(config["feature_space"], sample_frac=0.001 if QUICK_TEST else 1.0)

    # Run clustering
    labels, metadata = run_clustering_algorithm(X, config)

    # Compute clustering metrics
    silhouette, dbi = compute_clustering_metrics(X, labels)
    metadata.update({
        "run_id": run_id,
        "silhouette_score": silhouette,
        "dbi_score": dbi,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "n_noise": int(np.sum(labels == -1)) if -1 in labels else 0,
    })

    # Save outputs
    save_clustering_outputs(
        labels=labels,
        config=config,
        metadata=metadata,
        output_dir=RESULTS_DIR,
        label_dir=LABELS_DIR
    )

    end = datetime.now()
    duration = (end - start).total_seconds() / 60
    log_file.write(f"[{end.strftime('%H:%M:%S')}] Completed {run_id} in {duration:.2f} min\n")
    print(f"[{end.strftime('%H:%M:%S')}] Done: {run_id} ({duration:.2f} min)")

if __name__ == "__main__":
    configs = load_clustering_configs()

    with open(LOG_FILE, "w") as log_file:
        log_file.write("=== Unsupervised Clustering Run Log ===\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for config in configs:
            try:
                run_clustering_job(config, log_file)
            except Exception as e:
                log_file.write(f"[ERROR] {config}: {str(e)}\n")
                print(f"[ERROR] {config}: {str(e)}")

        log_file.write(f"\nCompleted all jobs at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(" All clustering jobs completed.")
