import json

import torch

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN


def extract_embeddings(model_dir, layer_index=-1):
    """
    Extracts node embeddings from a specific layer of the trained GNN.

    Args:
        model_dir (str): Directory containing the model's .pth file and metadata.json.
        layer_index (int): Index of the layer to extract embeddings from (-1 for the last layer).

    Returns:
        torch.Tensor: Node embeddings.
        torch.Tensor: Node labels.
    """
    # Load metadata
    metadata_path = f"{model_dir}/metadata.json"
    model_path = f"{model_dir}/model.pth"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load dataset
    dataset_name = metadata["dataset"]["name"]
    dataset = DatasetLoader(dataset_name).load()
    data = dataset[0]

    # Move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

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
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Extract embeddings
    hooks = []
    embeddings = []

    def hook_fn(module, input, output):
        embeddings.append(output)

    hooks.append(model.layers[layer_index].register_forward_hook(hook_fn))

    with torch.no_grad():
        model(data)

    for hook in hooks:
        hook.remove()

    return embeddings[0].cpu(), data.y.cpu()
