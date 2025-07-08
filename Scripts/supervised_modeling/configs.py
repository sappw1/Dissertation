import json
import os


def load_experiment_configs(mode="gpu"):  # mode = "gpu" or "cpu"
    base_models = [
        {"model_name": "logreg", "apply_smote": True, "use_gpu": False},
        {"model_name": "svm", "apply_smote": True, "use_gpu": True},
        {"model_name": "naive_bayes", "apply_smote": True, "use_gpu": True},
        {"model_name": "neural_net", "apply_smote": True, "use_gpu": True},
        {"model_name": "rf", "apply_smote": True, "use_gpu": True},
        {"model_name": "xgboost", "apply_smote": True, "use_gpu": True}
    ]

    # Filter based on mode
    if mode == "gpu":
        base_models = [m for m in base_models if m["use_gpu"]]
    elif mode == "cpu":
        base_models = [m for m in base_models if not m["use_gpu"]]

    config_path = "/content/drive/MyDrive/NCU/Dissertation/Scripts/supervised_modeling/model_config_all_features.json"
    with open(config_path, "r") as f:
        cluster_configs = json.load(f)

    experiment_configs = []
    for model in base_models:
        for cluster in cluster_configs:
            cfg = {
                "model_name": model["model_name"],
                "apply_smote": model["apply_smote"],
                "use_gpu": model["use_gpu"],
                "cluster_condition": cluster["name"],
                "cluster_features": cluster["cluster_features"]
            }
            experiment_configs.append(cfg)

    return experiment_configs
