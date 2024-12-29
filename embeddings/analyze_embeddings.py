import json

import torch

from data.dataset_loader import DatasetLoader
from utils.load_generalized_gnn import load_generalized_gnn


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
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load dataset
    dataset_name = metadata["dataset"]["name"]
    dataset = DatasetLoader(dataset_name).load()
    data = dataset[0]

    # Initialize model
    model = load_generalized_gnn(model_dir)

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

    return embeddings[0], data.y
