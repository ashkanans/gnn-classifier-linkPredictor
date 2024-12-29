import torch
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_


class GNNExplainability:
    def __init__(self, model, data, epochs=100, lr=0.01):
        """
        Initialize the GNNExplainer for explainability.

        Args:
            model (GeneralizedGNN): Trained GeneralizedGNN model.
            data (torch_geometric.data.Data): Graph data.
            epochs (int): Number of epochs for the explainer to train.
            lr (float): Learning rate for the explainer.
        """
        self.model = model
        self.data = data
        self.explainer = GNNExplainer_(
            model=self.model,
            epochs=epochs,
            lr=lr,
            feat_mask_type="feature",  # Masking features at the node level
            allow_edge_mask=True  # Enable edge masking
        )

    def explain_node(self, node_idx):
        """
        Explain the prediction for a specific node.

        Args:
            node_idx (int): Index of the node to explain.

        Returns:
            dict: Explanation results containing important edges, features, etc.
        """
        print(f"Explaining node {node_idx}...")

        # Explain the node prediction using GNNExplainer_
        node_feat_mask, edge_mask = self.explainer.explain_node(
            node_idx=node_idx,
            x=self.data.x,
            edge_index=self.data.edge_index
        )

        # Get the model's prediction for the node using the entire data object
        prediction = self.model(data=self.data).argmax(dim=1)[node_idx]

        print(f"Node {node_idx} is predicted as class {prediction.item()}.")

        return {
            "node_feat_mask": node_feat_mask,
            "edge_mask": edge_mask,
            "prediction": prediction
        }

    def visualize_node(self, node_idx, edge_mask, path=None):
        """
        Visualize the explanation for a specific node.

        Args:
            node_idx (int): Index of the node to visualize.
            edge_mask (torch.Tensor): Importance mask for edges.
            path (str): Path to save the plot (optional).
        """
        from torch_geometric.utils import to_networkx
        import networkx as nx
        import matplotlib.pyplot as plt

        # Convert the graph to a NetworkX graph
        G = to_networkx(self.data, to_undirected=True)

        # Highlight important edges with a simple threshold-based binary color
        edge_weights = edge_mask.detach().numpy()
        edge_colors = ['blue' if w > 0.5 else 'lightgray' for w in edge_weights]

        # Node colors based on labels (if available)
        if hasattr(self.data, 'y'):
            node_colors = ['red' if label == self.data.y[node_idx].item() else 'gray' for label in self.data.y]
        else:
            node_colors = 'lightgray'

        # Position the nodes using a spring layout
        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 8))

        # Draw the graph with simple colors
        nx.draw(
            G, pos,
            with_labels=True,
            edge_color=edge_colors,
            node_color=node_colors,
            node_size=500,
            font_size=10
        )

        # Save or display the plot
        if path:
            plt.savefig(path)
        else:
            plt.show(block=True)

    def explain_all_nodes(self, num_nodes=5):
        """
        Explain predictions for a subset of nodes.

        Args:
            num_nodes (int): Number of nodes to explain.
        """
        explained_nodes = torch.randperm(self.data.num_nodes)[:num_nodes]
        for node_idx in explained_nodes:
            explanation = self.explain_node(node_idx)
            print(f"Explanation for node {node_idx}:")
            print(f"  Edge Mask: {explanation['edge_mask']}")
            print(f"  Node Feature Mask: {explanation['node_feat_mask']}")
            self.visualize_node(node_idx, explanation["edge_mask"])
