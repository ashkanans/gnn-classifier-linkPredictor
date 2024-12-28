import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.utils import negative_sampling

from models.link_prediction_gnn import GeneralizedGNNForEdgePrediction


class EdgePredictionTrainer:
    def __init__(self, dataset, model_config, lr=0.01, weight_decay=5e-4, val_split=0.1):
        """
        Initializes the trainer for the GeneralizedGNNForEdgePrediction model.

        :param dataset: The PyTorch Geometric dataset containing the graph data.
        :param model_config: Dictionary containing model configurations (e.g., input_dim, hidden_dim).
        :param lr: Learning rate for the optimizer.
        :param weight_decay: Weight decay (L2 regularization) for the optimizer.
        :param val_split: Fraction of edges to use for validation.
        """
        self.data = dataset[0]

        # Split edges into training and validation sets
        self.train_edge_index, self.val_edge_index = self.split_edges(self.data.edge_index, val_split)

        self.model = GeneralizedGNNForEdgePrediction(**model_config)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def split_edges(self, edge_index, val_split):
        """
        Splits edges into training and validation sets.

        :param edge_index: Full edge index (torch.Tensor of shape [2, num_edges]).
        :param val_split: Fraction of edges to use for validation.
        :return: Training edge index, validation edge index.
        """
        num_edges = edge_index.size(1)
        num_val = int(num_edges * val_split)
        perm = torch.randperm(num_edges)

        val_edge_index = edge_index[:, perm[:num_val]]
        train_edge_index = edge_index[:, perm[num_val:]]

        return train_edge_index, val_edge_index

    def train(self, epochs=100):
        """
        Train the GNN model for edge-level predictions.

        :param epochs: Number of training epochs.
        """
        self.model.train()
        best_val_auc = 0
        best_model_state = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Generate negative edges dynamically
            neg_train_edge_index = negative_sampling(
                edge_index=self.train_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.train_edge_index.size(1),
            )

            # Prepare training labels
            pos_train_labels = torch.ones(self.train_edge_index.size(1))
            neg_train_labels = torch.zeros(neg_train_edge_index.size(1))

            # Concatenate positive and negative training edges
            combined_train_edge_index = torch.cat([self.train_edge_index, neg_train_edge_index], dim=1)
            combined_train_labels = torch.cat([pos_train_labels, neg_train_labels], dim=0)

            # Compute edge scores using the model
            self.data.edge_index = combined_train_edge_index  # Update the edge index
            train_edge_scores = self.model(self.data)  # Use model's forward method to compute edge scores

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(train_edge_scores, combined_train_labels)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Validation
            val_auc = self.validate()
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")

        # Load the best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)

    def validate(self):
        """
        Validate the model on the validation set and compute ROC-AUC.

        :return: Validation ROC-AUC.
        """
        self.model.eval()
        with torch.no_grad():
            # Generate negative validation edges
            neg_val_edge_index = negative_sampling(
                edge_index=self.val_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.val_edge_index.size(1),
            )

            # Prepare validation labels
            pos_val_labels = torch.ones(self.val_edge_index.size(1))
            neg_val_labels = torch.zeros(neg_val_edge_index.size(1))

            # Concatenate positive and negative validation edges
            combined_val_edge_index = torch.cat([self.val_edge_index, neg_val_edge_index], dim=1)
            combined_val_labels = torch.cat([pos_val_labels, neg_val_labels], dim=0)

            # Update data with the validation edge index
            original_edge_index = self.data.edge_index  # Save the original edge index
            self.data.edge_index = combined_val_edge_index  # Use validation edges for prediction

            # Compute validation predictions for combined_val_edge_index
            val_edge_scores = self.model(self.data)  # Predict edge scores for the validation edges
            val_edge_scores = torch.sigmoid(val_edge_scores)  # Convert logits to probabilities

            # Restore the original edge index
            self.data.edge_index = original_edge_index

            # Ensure shapes match
            assert val_edge_scores.shape[0] == combined_val_labels.shape[0], \
                f"Mismatch: val_edge_scores shape {val_edge_scores.shape} vs combined_val_labels shape {combined_val_labels.shape}"

            # Compute ROC-AUC
            val_auc = roc_auc_score(combined_val_labels.cpu().numpy(), val_edge_scores.cpu().numpy())

        return val_auc

    def evaluate(self, edge_index=None, labels=None):
        """
        Evaluate the model on a validation or test set.

        :param edge_index: Edge indices for evaluation (torch.LongTensor of shape [2, num_edges]).
        :param labels: Ground truth labels for the edges (torch.Tensor of shape [num_edges]).
        :return: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            # Use provided edge_index and labels, or generate them if not provided
            if edge_index is None or labels is None:
                # Perform negative sampling and create test edge set
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=self.data.edge_index.size(1),
                )
                edge_index = torch.cat([self.data.edge_index, neg_edge_index], dim=1)
                pos_labels = torch.ones(self.data.edge_index.size(1))
                neg_labels = torch.zeros(neg_edge_index.size(1))
                labels = torch.cat([pos_labels, neg_labels], dim=0)

            # Temporarily replace edge_index in self.data for prediction
            original_edge_index = self.data.edge_index
            self.data.edge_index = edge_index
            edge_scores = self.model(self.data)  # Compute edge scores
            self.data.edge_index = original_edge_index  # Restore the original edge_index

            # Convert logits to probabilities
            predictions = torch.sigmoid(edge_scores)

            # Ensure shapes match
            assert predictions.shape[0] == labels.shape[0], \
                f"Mismatch: predictions shape {predictions.shape} vs labels shape {labels.shape}"

            # Compute binary predictions
            threshold = 0.5
            predictions_binary = (predictions > threshold).int()

            # Convert tensors to NumPy arrays for metrics computation
            labels_np = labels.cpu().numpy()  # Ensure it's on CPU before converting to NumPy
            predictions_binary_np = predictions_binary.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            # Compute evaluation metrics
            metrics = {
                "accuracy": accuracy_score(labels_np, predictions_binary_np),
                "precision": precision_score(labels_np, predictions_binary_np, zero_division=0),
                "recall": recall_score(labels_np, predictions_binary_np, zero_division=0),
                "f1_score": f1_score(labels_np, predictions_binary_np, zero_division=0),
                "roc_auc": roc_auc_score(labels_np, predictions_np),
            }

            # Print metrics
            print(f"Evaluation Metrics: {metrics}")

            return metrics
