# utils.py
import os
import json
import numpy as np

# Paths to preprocessed inputs
DATA_PATH = "/content/drive/MyDrive/NCU/Dissertation/Data/Processed/Clustering/PCA"

def load_feature_data(feature_space, sample_frac=1.0):
    """
    Loads either the full dataset (no PCA) or PCA-reduced version.

    Args:
        feature_space (str): either 'pca' or 'no_pca'
    
    Returns:
        ndarray (NumPy or CuPy depending on downstream use)
    """
    if feature_space == "pca":
        file_path = os.path.join(DATA_PATH, "X_all_pca_3.npy")
    elif feature_space == "no_pca":
        file_path = os.path.join(DATA_PATH, "X_all_no_pca.npy")
    else:
        raise ValueError(f"Invalid feature_space: {feature_space}")
    
    X = np.load(file_path)

    if sample_frac < 1.0:
        n = int(len(X) * sample_frac)
        idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
        X = X[idx]

    return X

def save_clustering_outputs(labels, config, metadata, output_dir, label_dir):
    """
    Saves labels, metadata, and noise indices (if DBSCAN).
    """
    label_name = config.get("label_name", "default")
    
    # Save label array
    label_path = os.path.join(label_dir, f"{label_name}_labels.npy")
    np.save(label_path, labels)

    # Save noise indices for DBSCAN
    if config["algorithm"].lower() == "dbscan":
        noise_idx = np.where(labels == -1)[0]
        noise_path = os.path.join(label_dir, f"{label_name}_noise_indices.txt")
        with open(noise_path, "w") as f:
            for idx in noise_idx:
                f.write(f"{idx}\n")

    # Save JSON metadata
    result_path = os.path.join(output_dir, f"{label_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f" Saved: {label_name}")
