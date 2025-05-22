import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Global paths
BASE_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering"
CLUSTER_PATH = os.path.join(BASE_PATH, "cluster_features")
OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised"

def load_data(cluster_features=None, sample_frac=1.0):
    """
    Loads the full dataset and fraud labels, optionally selecting cluster features.

    Parameters:
    - cluster_features (list or None): 
        - None: load all cluster features
        - [] (empty list): drop all cluster features
        - list of str: include only specified cluster feature columns
    """
    X = pd.read_pickle(os.path.join(BASE_PATH, "X_all_augmented.pkl"))
    y = pd.read_pickle(os.path.join(BASE_PATH, "y_labels.pkl"))

    # Subsample
    if sample_frac < 1.0:
        sampled_idx = X.sample(frac=sample_frac, random_state=42).index
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    all_cluster_cols = [
    "kmeans_full_n2",
    "hier_full_n2",
    "dbscan_full_e07_m10_noise"
    ]


    if cluster_features is None:
        # Use all cluster features (do nothing)
        pass
    elif len(cluster_features) == 0:
        # Drop all cluster features
        X = X.drop(columns=[col for col in all_cluster_cols if col in X.columns])
    else:
        # Keep only specified cluster features
        drop_cols = [col for col in all_cluster_cols if col not in cluster_features]
        X = X.drop(columns=drop_cols)

    return X, y

def prepare_data(X, y, apply_smote=True, test_size=0.3, random_state=42):
    """
    Split and optionally apply SMOTE to training data.
    Ensures float32 dtype for GPU compatibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # Ensure all inputs are float32 for cuML compatibility
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, X_test, y_train, y_test


def save_results(results, config):
    """
    Save results to a JSON file using metadata from the result payload.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = results["model"]
    cluster_name = results.get("cluster_condition", "Baseline").replace(" ", "_")
    seed = results.get("seed", 0)

    run_id = f"{model}__{cluster_name}__seed{seed}"
    path = os.path.join(OUTPUT_DIR, run_id + ".json")

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f" Saved results to {path}")
