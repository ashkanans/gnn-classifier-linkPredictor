import json
import os


def find_best_models(saved_models_dir):
    # Initialize variables to track the best models and their accuracies
    best_pubmed_model = {"name": None, "accuracy": 0.0}
    best_citeseer_model = {"name": None, "accuracy": 0.0}
    best_cora_model = {"name": None, "accuracy": 0.0}

    # Iterate through each folder in the saved_models directory
    for folder in os.listdir(saved_models_dir):
        folder_path = os.path.join(saved_models_dir, folder)
        metadata_path = os.path.join(folder_path, "metadata.json")

        # Check if the metadata.json file exists
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Extract accuracies
            model_name = folder
            pubmed_accuracy = metadata.get("cross_test", {}).get("PubMed", {}).get("metrics", {}).get("accuracy", 0.0)
            citeseer_accuracy = metadata.get("cross_test", {}).get("CiteSeer", {}).get("metrics", {}).get("accuracy",
                                                                                                          0.0)
            cora_accuracy = metadata.get("evaluation", {}).get("accuracy", 0.0)

            # Update the best PubMed model
            if pubmed_accuracy > best_pubmed_model["accuracy"]:
                best_pubmed_model["name"] = model_name
                best_pubmed_model["accuracy"] = pubmed_accuracy

            # Update the best CiteSeer model
            if citeseer_accuracy > best_citeseer_model["accuracy"]:
                best_citeseer_model["name"] = model_name
                best_citeseer_model["accuracy"] = citeseer_accuracy

            # Update the best Cora model
            if cora_accuracy > best_cora_model["accuracy"]:
                best_cora_model["name"] = model_name
                best_cora_model["accuracy"] = cora_accuracy

    return best_pubmed_model, best_citeseer_model, best_cora_model
