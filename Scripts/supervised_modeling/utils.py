import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Global paths
BASE_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed"
CLUSTER_PATH = os.path.join(BASE_PATH, "cluster_features")
OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised"

def build_dataset_with_clusters(cluster_features=None):
    """
    Load base dataset and add selected cluster feature columns.
    """
    # Load base data
    X = pd.read_pickle(os.path.join(BASE_PATH, "X_all_scaled.pkl"))
    y = pd.read_pickle(os.path.join(BASE_PATH, "y_labels.pkl"))

    # Add cluster features if specified
    if cluster_features:
        for feat in cluster_features:
            feat_path = os.path.join(CLUSTER_PATH, f"{feat}.pkl")
            if not os.path.exists(feat_path):
                print(f" Missing cluster feature: {feat_path}")
                continue
            cluster_col = pd.read_pickle(feat_path)
            X[feat] = cluster_col
    return X, y

def prepare_data(X, y, apply_smote=True, test_size=0.3, random_state=42):
    """
    Split and optionally apply SMOTE to training data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

def save_results(results, model_config, cluster_config):
    """
    Save results to a JSON file with model and cluster metadata.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = model_config['model_name']
    cluster_name = cluster_config.get("name", "None").replace(" ", "_")
    run_id = f"{model}__{cluster_name}"
    path = os.path.join(OUTPUT_DIR, run_id + ".json")

    output = {
        "model_config": model_config,
        "cluster_config": cluster_config,
        "metrics": results
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=4)
    print(f" Saved results to {path}")
