import json

import torch

from models.generalized_gnn import GeneralizedGNN


def load_generalized_gnn(model_dir):
    """
    Creates and returns a GeneralizedGNN model initialized with the metadata from the given model directory.

    Args:
        model_dir (str): Directory containing the model's .pth file and metadata.json.

    Returns:
        GeneralizedGNN: The initialized GeneralizedGNN model.
    """
    # Load metadata
    metadata_path = f"{model_dir}/metadata.json"
    model_path = f"{model_dir}/model.pth"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize model based on metadata
    model = GeneralizedGNN(
        input_dim=metadata["dataset"]["num_features"],
        hidden_dim=metadata["hidden_dim"],
        output_dim=metadata["dataset"]["num_classes"],
        num_layers=metadata["num_layers"],
        variant=metadata["variant"],
        dropout=metadata["dropout"],
        use_residual=metadata["use_residual"],
        use_layer_norm=metadata["use_layer_norm"]
    )

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model
