import os.path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.device import DeviceHandler
from utils.model_saver import ModelSaver


class GNNEvaluator:
    def __init__(self, save_dir, dataset_name="Cora", model_type="simple", hidden_dim=64, num_layers=2, variant="gcn",
                 dropout=0.5, use_residual=False, use_layer_norm=False):
        self.dataset = DatasetLoader(dataset_name).load()
        self.data = self.dataset[0]

        # Move data to the device
        self.device = DeviceHandler.get_device()
        self.data = DeviceHandler.move_data_to_device(self.data, self.device)

        self.save_dir = save_dir

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
        model_path = os.path.join(save_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluate(self):
        with torch.no_grad():
            logits = self.model(self.data)
            pred = logits[self.data.test_mask].argmax(dim=1)
            true = self.data.y[self.data.test_mask]

        metrics = self.compute_metrics(true, pred)
        print(metrics)

        ModelSaver.update_metadata(self.save_dir, {"evaluation": metrics})

    def compute_metrics(self, true, pred):
        return {
            "accuracy": accuracy_score(true.cpu(), pred.cpu()),
            "precision": precision_score(true.cpu(), pred.cpu(), average="macro", zero_division=0),
            "recall": recall_score(true.cpu(), pred.cpu(), average="macro", zero_division=0),
            "f1_score": f1_score(true.cpu(), pred.cpu(), average="macro", zero_division=0)
        }
