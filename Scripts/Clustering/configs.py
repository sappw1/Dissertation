# configs.py
import json
import os

CONFIG_PATH = "/content/drive/MyDrive/NCU/Dissertation/Scripts/Clustering/cluster_config.json"

def load_clustering_configs():
    with open(CONFIG_PATH, "r") as f:
        base_configs = json.load(f)

    expanded_configs = []

    for base in base_configs:
        algorithm = base["algorithm"]
        prefix = base["label_prefix"]
        feature_space = base["feature_space"]
        sweep = base["sweep"]
        param_name = sweep["param"]
        param_values = sweep["values"]

        # DBSCAN gets eps × min_samples sweep
        if algorithm.lower() == "dbscan":
            min_samples_list = sweep["min_samples"] if isinstance(sweep.get("min_samples"), list) else [sweep.get("min_samples", 5)]

            for min_samples in min_samples_list:
                for val in param_values:
                    config = {
                        "algorithm": algorithm,
                        "feature_space": feature_space,
                        "params": {
                            param_name: val,
                            "min_samples": min_samples
                        },
                        "label_name": f"{prefix}_{param_name[0]}{val}_m{min_samples}"
                    }
                    expanded_configs.append(config)

        else:
            # For kmeans, hierarchical, etc.
            for val in param_values:
                config = {
                    "algorithm": algorithm,
                    "feature_space": feature_space,
                    "params": {param_name: val},
                    "label_name": f"{prefix}_{param_name[0]}{val}"
                }
                expanded_configs.append(config)

    return expanded_configs

