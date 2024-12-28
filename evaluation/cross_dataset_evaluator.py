import os.path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.device import DeviceHandler  # Import the DeviceHandler
from utils.dimensionality_handler import zero_pad_features, replicate_features, DimensionalityReducer
from utils.model_saver import ModelSaver


class CrossDatasetEvaluator:
    def __init__(
            self,
            save_dir,
            config,
            datasets=["CiteSeer", "PubMed"],
            default_handling="auto",
            model_type="simple",
            hidden_dim=64,
            num_layers=2,
            variant="gcn",
            dropout=0.5,
            use_residual=False,
            use_layer_norm=False,
            model_path="models/gnn_model.pth",
    ):
        self.config = config
        self.datasets = datasets
        self.default_handling = default_handling
        self.save_dir = save_dir

        # Load the appropriate model based on the model_type
        if model_type == "simple":
            self.model = GNNModel(
                input_dim=config.INPUT_DIM,
                hidden_dim=hidden_dim,
                output_dim=config.OUTPUT_DIM,
            )
        elif model_type == "generalized":
            self.model = GeneralizedGNN(
                input_dim=config.INPUT_DIM,
                hidden_dim=hidden_dim,
                output_dim=config.OUTPUT_DIM,
                num_layers=num_layers,
                variant=variant,
                dropout=dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move model to the appropriate device and load weights
        self.model, self.device = DeviceHandler.move_model_to_device(self.model)
        model_path = os.path.join(save_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluate(self):
        cross_test_results = {}
        for dataset_name in self.datasets:
            print(f"\nEvaluating on {dataset_name} dataset:")
            dataset = DatasetLoader(dataset_name).load()
            data = dataset[0]

            # Move data to the device
            data = DeviceHandler.move_data_to_device(data, self.device)

            # Handle feature dimension differences
            current_dim = data.x.shape[1]
            target_dim = self.config.INPUT_DIM

            handling_technique = "Not needed. Dimensions matched"
            if current_dim < target_dim:
                print(
                    f"Feature dimension mismatch: Model expects {target_dim}, but dataset has {current_dim} features."
                )
                handling_technique = self.handle_lower_dimensions(data, target_dim)
                print(f"Features were adjusted using '{self.default_handling}' to match the model's input dimension.")

            elif current_dim > target_dim:
                print(
                    f"Feature dimension mismatch: Model expects {target_dim}, but dataset has {current_dim} features."
                )
                handling_technique = self.handle_higher_dimensions(data, target_dim)
                print(f"Features were reduced using '{self.default_handling}' to match the model's input dimension.")

            # Perform inference
            with torch.no_grad():
                logits = self.model(data)
                pred = logits[data.test_mask].argmax(dim=1)
                true = data.y[data.test_mask]

            metrics = self.compute_metrics(true, pred)
            cross_test_results[dataset_name] = {
                "metrics": metrics,
                "dataset_info": {
                    "num_nodes": data.x.shape[0],
                    "num_features": current_dim,
                    "num_classes": len(data.y.unique())
                },
                "dimension_handling": handling_technique
            }

        ModelSaver.update_metadata(self.save_dir, {"cross_test": cross_test_results})

    def handle_lower_dimensions(self, data, target_dim):
        if self.default_handling == "auto" or self.default_handling == "zero_pad":
            print(f"Zero-padding features to match {target_dim} dimensions...")
            data.x = zero_pad_features(data.x, target_dim)
            return "zero_pad"
        elif self.default_handling == "replicate":
            print(f"Replicating features to match {target_dim} dimensions...")
            data.x = replicate_features(data.x, target_dim)
            return "replicate"
        else:
            raise ValueError(f"Unsupported handling method for lower dimensions: {self.default_handling}")

    def handle_higher_dimensions(self, data, target_dim):
        if self.default_handling == "auto" or self.default_handling == "pca":
            print(f"Reducing features using PCA to match {target_dim} dimensions...")
            reducer = DimensionalityReducer(target_dim)
            data.x = reducer.fit_transform(data.x)
            return "pca"
        else:
            raise ValueError(f"Unsupported handling method for higher dimensions: {self.default_handling}")

    def compute_metrics(self, true, pred):
        return {
            "accuracy": accuracy_score(true.cpu(), pred.cpu()),
            "precision": precision_score(true.cpu(), pred.cpu(), average="macro", zero_division=0),
            "recall": recall_score(true.cpu(), pred.cpu(), average="macro", zero_division=0),
            "f1_score": f1_score(true.cpu(), pred.cpu(), average="macro", zero_division=0)
        }
