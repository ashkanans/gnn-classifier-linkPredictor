import torch
from torch_geometric.utils import negative_sampling


def prepare_link_prediction_data(data):
    """
    Prepare data for link prediction by generating edge_label_index and edge_label.

    Args:
        data (torch_geometric.data.Data): Input graph data.

    Returns:
        torch_geometric.data.Data: Updated data with edge_label_index and edge_label.
    """
    if not hasattr(data, 'edge_label_index') or not hasattr(data, 'edge_label'):
        # Use the existing edges as positive edges
        pos_edge_index = data.edge_index

        # Generate negative edges using negative sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)  # Match the number of positive edges
        )

        # Combine positive and negative edges
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

        # Create edge labels (1 for positive, 0 for negative)
        edge_label = torch.cat([
            torch.ones(pos_edge_index.size(1)),  # Positive edges
            torch.zeros(neg_edge_index.size(1))  # Negative edges
        ], dim=0)

        # Add to the data object
        data.edge_label_index = edge_label_index
        data.edge_label = edge_label

    return data
