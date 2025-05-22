# metrics.py
import numpy as np
import cupy as cp
from sklearn.metrics import davies_bouldin_score
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

def compute_clustering_metrics(X, labels, max_clusters=50):
    """
    Compute silhouette and Davies-Bouldin Index (DBI).
    Automatically skip metrics if clustering is not valid.

    Returns:
    - silhouette_score (float or None)
    - dbi_score (float or None)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters < 2 or n_clusters > max_clusters:
        return None, None  # Metrics not valid or not meaningful

    # Ensure data is on GPU for silhouette
    if not isinstance(X, cp.ndarray):
        X_gpu = cp.asarray(X)
    else:
        X_gpu = X

    try:
        sil_score = float(cython_silhouette_score(X_gpu, labels))
    except Exception as e:
        print(f"[Warning] Silhouette score failed: {e}")
        sil_score = None

    try:
        dbi_score = davies_bouldin_score(np.asarray(X), labels)
    except Exception as e:
        print(f"[Warning] DBI score failed: {e}")
        dbi_score = None

    return sil_score, dbi_score
