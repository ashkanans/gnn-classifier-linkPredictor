import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LinkPredictionEvaluator:
    def __init__(self, model, decoder, dataset_loader):
        """
        Initialize the Link Prediction Evaluator.

        :param model: The GNN model to evaluate.
        :param decoder: The link prediction decoder.
        :param dataset_loader: An instance of DatasetLoader with prepared data.
        """
        self.model = model
        self.decoder = decoder
        self.dataset_loader = dataset_loader

    def evaluate(self):
        """
        Evaluate the link prediction model on the test dataset.
        """
        self.model.eval()

        # Retrieve test edges and labels
        edge_index, labels = self.dataset_loader.get_test_edges()
        pos_edge_index = edge_index[:, labels == 1]
        neg_edge_index = edge_index[:, labels == 0]

        with torch.no_grad():
            # Get node embeddings
            z = self.model(self.dataset_loader.data)

            # Compute logits for positive and negative edges
            pos_logits = self.decoder(z, pos_edge_index)
            neg_logits = self.decoder(z, neg_edge_index)

            # Combine logits and labels
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            all_labels = torch.cat([torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0)

        # Calculate metrics directly (no explicit move to CPU as we're already using CPU by default)
        accuracy = accuracy_score(all_labels.numpy(), (logits.numpy() > 0).astype(int))
        precision = precision_score(all_labels.numpy(), (logits.numpy() > 0).astype(int))
        recall = recall_score(all_labels.numpy(), (logits.numpy() > 0).astype(int))
        f1 = f1_score(all_labels.numpy(), (logits.numpy() > 0).astype(int))
        roc_auc = roc_auc_score(all_labels.numpy(), logits.numpy())

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        # Return metrics for further analysis
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
