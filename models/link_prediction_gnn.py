import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GeneralizedGNNForEdgePrediction(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=2,
            variant="gcn",  # Options: 'gcn', 'sage', 'gat'
            dropout=0.5,
            use_residual=False,
            use_layer_norm=False,
    ):
        super(GeneralizedGNNForEdgePrediction, self).__init__()

        self.variant = variant
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

        # Select the convolution layer based on the variant
        if variant == "gcn":
            ConvLayer = GCNConv
        elif variant == "sage":
            ConvLayer = SAGEConv
        elif variant == "gat":
            ConvLayer = GATConv
        else:
            raise ValueError(f"Unsupported GNN variant: {variant}")

        # Create the initial layer
        self.layers = nn.ModuleList()
        self.layers.append(ConvLayer(input_dim, hidden_dim))

        # Create the hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(ConvLayer(hidden_dim, hidden_dim))

        # Create the output layer
        self.layers.append(ConvLayer(hidden_dim, output_dim))

        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])

        # Residual connections
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        # Edge prediction layer
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single score per edge
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Initial input projection for residual connections
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x

        # Pass through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.use_residual:
                x = x + residual  # Add residual connection
                residual = x  # Update residual for the next layer

        # Output layer (final node embeddings)
        x = self.layers[-1](x, edge_index)  # Shape: [num_nodes, output_dim]

        # Compute edge-level embeddings
        edge_src, edge_dst = edge_index
        edge_src_embeddings = x[edge_src]  # Shape: [num_edges, output_dim]
        edge_dst_embeddings = x[edge_dst]  # Shape: [num_edges, output_dim]
        edge_features = torch.cat([edge_src_embeddings, edge_dst_embeddings],
                                  dim=-1)  # Shape: [num_edges, 2 * output_dim]

        # Ensure edge_features is properly shaped
        assert edge_features.dim() == 2, f"edge_features should be 2D, got {edge_features.dim()}D"
        assert edge_features.size(1) == 64, f"edge_features second dimension should be 64, got {edge_features.size(1)}"

        # Compute edge scores
        edge_scores = self.edge_predictor(edge_features).squeeze(-1)  # Shape: [num_edges]

        return edge_scores
