import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Global paths
BASE_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering"
CLUSTER_PATH = os.path.join(BASE_PATH, "cluster_features")
OUTPUT_DIR = "/content/drive/MyDrive/NCU/Dissertation/Results/Supervised/PCA_run/all_features"

def load_data(cluster_features=None, sample_frac=1.0):
    """
    Loads the full dataset and fraud labels, optionally selecting cluster features.

    Parameters:
    - cluster_features (list or None): 
        - None: load all cluster features
        - [] (empty list): drop all cluster features
        - list of str: include only specified cluster feature columns
    """
    X = pd.read_pickle(os.path.join(BASE_PATH, "X_pca_augmented_all_clusters.pkl"))
    y = pd.read_pickle(os.path.join(BASE_PATH, "y_labels.pkl"))

    # Subsample
    if sample_frac < 1.0:
        sampled_idx = X.sample(frac=sample_frac, random_state=42).index
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    all_cluster_cols = ['kmeans_full_n2_labels', 'kmeans_full_n3_labels', 'kmeans_full_n4_labels', 'kmeans_full_n5_labels', 
                        'kmeans_full_n6_labels', 'kmeans_full_n7_labels', 'kmeans_full_n8_labels', 'kmeans_full_n9_labels', 
                        'kmeans_full_n10_labels', 'kmeans_pca_n2_labels', 'kmeans_pca_n3_labels', 'kmeans_pca_n4_labels', 
                        'kmeans_pca_n5_labels', 'kmeans_pca_n6_labels', 'kmeans_pca_n7_labels', 'kmeans_pca_n8_labels', 
                        'kmeans_pca_n9_labels', 'kmeans_pca_n10_labels', 'dbscan_pca_e0.3_m3_labels', 'dbscan_pca_e0.3_m3_noise_indices', 
                        'dbscan_pca_e0.5_m3_labels', 'dbscan_pca_e0.5_m3_noise_indices', 'dbscan_pca_e0.7_m3_labels', 
                        'dbscan_pca_e0.7_m3_noise_indices', 'dbscan_pca_e1.0_m3_labels', 'dbscan_pca_e1.0_m3_noise_indices', 
                        'dbscan_pca_e1.3_m3_labels', 'dbscan_pca_e1.3_m3_noise_indices', 'dbscan_pca_e0.3_m5_labels', 
                        'dbscan_pca_e0.3_m5_noise_indices', 'dbscan_pca_e0.5_m5_labels', 'dbscan_pca_e0.5_m5_noise_indices', 
                        'dbscan_pca_e0.7_m5_labels', 'dbscan_pca_e0.7_m5_noise_indices', 'dbscan_pca_e1.0_m5_labels', 
                        'dbscan_pca_e1.0_m5_noise_indices', 'dbscan_pca_e1.3_m5_labels', 'dbscan_pca_e1.3_m5_noise_indices', 
                        'dbscan_pca_e0.3_m7_labels', 'dbscan_pca_e0.3_m7_noise_indices', 'dbscan_pca_e0.5_m7_labels', 
                        'dbscan_pca_e0.5_m7_noise_indices', 'dbscan_pca_e0.7_m7_labels', 'dbscan_pca_e0.7_m7_noise_indices', 
                        'dbscan_pca_e1.0_m7_labels', 'dbscan_pca_e1.0_m7_noise_indices', 'dbscan_pca_e1.3_m7_labels', 
                        'dbscan_pca_e1.3_m7_noise_indices', 'dbscan_pca_e0.3_m10_labels', 'dbscan_pca_e0.3_m10_noise_indices', 
                        'dbscan_pca_e0.5_m10_labels', 'dbscan_pca_e0.5_m10_noise_indices', 'dbscan_pca_e0.7_m10_labels', 
                        'dbscan_pca_e0.7_m10_noise_indices', 'dbscan_pca_e1.0_m10_labels', 'dbscan_pca_e1.0_m10_noise_indices', 
                        'dbscan_pca_e1.3_m10_labels', 'dbscan_pca_e1.3_m10_noise_indices', 'dbscan_pca_e0.3_m15_labels', 
                        'dbscan_pca_e0.3_m15_noise_indices', 'dbscan_pca_e0.5_m15_labels', 'dbscan_pca_e0.5_m15_noise_indices', 
                        'dbscan_pca_e0.7_m15_labels', 'dbscan_pca_e0.7_m15_noise_indices', 'dbscan_pca_e1.0_m15_labels', 
                        'dbscan_pca_e1.0_m15_noise_indices', 'dbscan_pca_e1.3_m15_labels', 'dbscan_pca_e1.3_m15_noise_indices', 
                        'dbscan_full_e0.3_m3_labels', 'dbscan_full_e0.3_m3_noise_indices', 'dbscan_full_e0.5_m3_labels', 
                        'dbscan_full_e0.5_m3_noise_indices', 'dbscan_full_e0.7_m3_labels', 'dbscan_full_e0.7_m3_noise_indices', 
                        'dbscan_full_e1.0_m3_labels', 'dbscan_full_e1.0_m3_noise_indices', 'dbscan_full_e1.3_m3_labels', 
                        'dbscan_full_e1.3_m3_noise_indices', 'dbscan_full_e0.3_m5_labels', 'dbscan_full_e0.3_m5_noise_indices', 
                        'dbscan_full_e0.5_m5_labels', 'dbscan_full_e0.5_m5_noise_indices', 'dbscan_full_e0.7_m5_labels', 
                        'dbscan_full_e0.7_m5_noise_indices', 'dbscan_full_e1.0_m5_labels', 'dbscan_full_e1.0_m5_noise_indices', 
                        'dbscan_full_e1.3_m5_labels', 'dbscan_full_e1.3_m5_noise_indices', 'dbscan_full_e0.3_m7_labels', 
                        'dbscan_full_e0.3_m7_noise_indices', 'dbscan_full_e0.5_m7_labels', 'dbscan_full_e0.5_m7_noise_indices', 
                        'dbscan_full_e0.7_m7_labels', 'dbscan_full_e0.7_m7_noise_indices', 'dbscan_full_e1.0_m7_labels', 
                        'dbscan_full_e1.0_m7_noise_indices', 'dbscan_full_e1.3_m7_labels', 'dbscan_full_e1.3_m7_noise_indices', 
                        'dbscan_full_e0.3_m10_labels', 'dbscan_full_e0.3_m10_noise_indices', 'dbscan_full_e0.5_m10_labels', 
                        'dbscan_full_e0.5_m10_noise_indices', 'dbscan_full_e0.7_m10_labels', 'dbscan_full_e0.7_m10_noise_indices', 
                        'dbscan_full_e1.0_m10_labels', 'dbscan_full_e1.0_m10_noise_indices', 'dbscan_full_e1.3_m10_labels', 
                        'dbscan_full_e1.3_m10_noise_indices', 'dbscan_full_e0.3_m15_labels', 'dbscan_full_e0.3_m15_noise_indices', 
                        'dbscan_full_e0.5_m15_labels', 'dbscan_full_e0.5_m15_noise_indices', 'dbscan_full_e0.7_m15_labels', 
                        'dbscan_full_e0.7_m15_noise_indices', 'dbscan_full_e1.0_m15_labels', 'dbscan_full_e1.0_m15_noise_indices', 
                        'dbscan_full_e1.3_m15_labels', 'dbscan_full_e1.3_m15_noise_indices', 'hier_full_n2_labels', 'hier_full_n3_labels', 
                        'hier_full_n4_labels', 'hier_full_n5_labels', 'hier_full_n6_labels', 'hier_pca_n2_labels', 'hier_pca_n3_labels', 
                        'hier_pca_n4_labels', 'hier_pca_n5_labels', 'hier_pca_n6_labels']


    if cluster_features is None:
        # Use all cluster features (do nothing)
        pass
    elif len(cluster_features) == 0:
        # Drop all cluster features
        X = X.drop(columns=[col for col in all_cluster_cols if col in X.columns])
        X.columns = X.columns.astype(str)
    else:
        # Keep only specified cluster features
        drop_cols = [col for col in all_cluster_cols if col not in cluster_features]
        X = X.drop(columns=drop_cols)
        X.columns = X.columns.astype(str)

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
