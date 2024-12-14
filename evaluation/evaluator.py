import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.device import DeviceHandler


class GNNEvaluator:
    def __init__(self, dataset_name="Cora", model_type="simple", hidden_dim=64, num_layers=2, variant="gcn",
                 dropout=0.5, use_residual=False, use_layer_norm=False, model_path="models/gnn_model.pth"):
        self.dataset = DatasetLoader(dataset_name).load()
        self.data = self.dataset[0]

        # Move data to the device
        self.device = DeviceHandler.get_device()
        self.data = DeviceHandler.move_data_to_device(self.data, self.device)

        # Load the appropriate model based on the model_type
        if model_type == "simple":
            self.model = GNNModel(input_dim=self.dataset.num_features, hidden_dim=hidden_dim,
                                  output_dim=self.dataset.num_classes)
        elif model_type == "generalized":
            self.model = GeneralizedGNN(input_dim=self.dataset.num_features, hidden_dim=hidden_dim,
                                        output_dim=self.dataset.num_classes, num_layers=num_layers, variant=variant,
                                        dropout=dropout, use_residual=use_residual, use_layer_norm=use_layer_norm)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move model to the device and load weights
        self.model, _ = DeviceHandler.move_model_to_device(self.model)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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
