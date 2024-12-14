import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel


class GNNEvaluator:
    def __init__(
            self,
            dataset_name="Cora",
            model_type="simple",
            hidden_dim=64,
            num_layers=2,
            variant="gcn",
            dropout=0.5,
            use_residual=False,
            use_layer_norm=False,
            model_path="models/gnn_model.pth",
    ):
        """
        Initialize the evaluator with the specified model type and parameters.

        :param dataset_name: The name of the dataset to load.
        :param model_type: "simple" for Simple GNN, "generalized" for Generalized GNN.
        :param hidden_dim: Dimension of hidden layers.
        :param num_layers: Number of layers for the generalized GNN.
        :param variant: Type of GNN variant ("gcn", "sage", "gat") for generalized GNN.
        :param dropout: Dropout rate for generalized GNN.
        :param use_residual: Boolean to enable residual connections.
        :param use_layer_norm: Boolean to enable layer normalization.
        :param model_path: Path to the saved model file.
        """
        self.dataset = DatasetLoader(dataset_name).load()
        self.data = self.dataset[0]

        # Load the appropriate model based on the model_type
        if model_type == "simple":
            self.model = GNNModel(
                input_dim=self.dataset.num_features,
                hidden_dim=hidden_dim,
                output_dim=self.dataset.num_classes,
            )
        elif model_type == "generalized":
            self.model = GeneralizedGNN(
                input_dim=self.dataset.num_features,
                hidden_dim=hidden_dim,
                output_dim=self.dataset.num_classes,
                num_layers=num_layers,
                variant=variant,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load the model weights
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def evaluate(self):
        with torch.no_grad():
            logits = self.model(self.data)
            pred = logits[self.data.test_mask].argmax(dim=1)
            true = self.data.y[self.data.test_mask]

        self.compute_metrics(true, pred)

    def compute_metrics(self, true, pred):
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average="macro", zero_division=0)
        recall = recall_score(true, pred, average="macro", zero_division=0)
        f1 = f1_score(true, pred, average="macro", zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
