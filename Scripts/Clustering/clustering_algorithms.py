# clustering_algorithms.py
import cuml
import cupy as cp
from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
from cuml.cluster import AgglomerativeClustering as cuAgglomerative
from cuml.common.exceptions import NotFittedError

def run_clustering_algorithm(X, config):
    algorithm = config["algorithm"].lower()
    params = config.get("params", {})
    metadata = {
        "algorithm": algorithm,
        "feature_space": config["feature_space"],
        "params": params,
        "label_name": config.get("label_name", "unnamed")
    }

    # Ensure input is on GPU
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)

    if algorithm == "kmeans":
        model = cuKMeans(n_clusters=params.get("n_clusters", 2), random_state=42, n_init="auto")
        labels = model.fit_predict(X)
    elif algorithm == "dbscan":
        model = cuDBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5),
            metric=params.get("metric", "euclidean")
        )
        labels = model.fit_predict(X)
    elif algorithm == "hierarchical":
        model = cuAgglomerative(
            n_clusters=params.get("n_clusters", 2),
            linkage="single"
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Convert CuPy back to NumPy for downstream compatibility
    labels = cp.asnumpy(labels)

    return labels, metadata
