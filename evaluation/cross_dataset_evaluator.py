import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.gnn_model import GNNModel
from utils.config import Config
from utils.dimensionality_handler import zero_pad_features, replicate_features, DimensionalityReducer


class CrossDatasetEvaluator:
    def __init__(self, config, datasets=["CiteSeer", "PubMed"], default_handling="auto"):
        self.config = config
        self.datasets = datasets
        self.default_handling = default_handling

        # Load trained model
        self.model = GNNModel(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
        )
        self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH))
        self.model.eval()

    def evaluate(self):
        for dataset_name in self.datasets:
            print(f"\nEvaluating on {dataset_name} dataset:")
            dataset = DatasetLoader(dataset_name).load()
            data = dataset[0]

            # Handle feature dimension differences
            current_dim = data.x.shape[1]
            target_dim = self.config.INPUT_DIM

            if current_dim < target_dim:
                print(
                    f"Feature dimension mismatch: Model expects {target_dim}, but dataset has {current_dim} features.")
                self.handle_lower_dimensions(data, target_dim)
                print(f"Features were adjusted using '{self.default_handling}' to match the model's input dimension.")

            elif current_dim > target_dim:
                print(
                    f"Feature dimension mismatch: Model expects {target_dim}, but dataset has {current_dim} features.")
                self.handle_higher_dimensions(data, target_dim)
                print(f"Features were reduced using '{self.default_handling}' to match the model's input dimension.")

            # Perform inference
            with torch.no_grad():
                logits = self.model(data)
                pred = logits[data.test_mask].argmax(dim=1)
                true = data.y[data.test_mask]

            self.compute_metrics(true, pred)

    def handle_lower_dimensions(self, data, target_dim):
        if self.default_handling == "auto" or self.default_handling == "zero_pad":
            print(f"Zero-padding features to match {target_dim} dimensions...")
            data.x = zero_pad_features(data.x, target_dim)
        elif self.default_handling == "replicate":
            print(f"Replicating features to match {target_dim} dimensions...")
            data.x = replicate_features(data.x, target_dim)
        else:
            raise ValueError(f"Unsupported handling method for lower dimensions: {self.default_handling}")

    def handle_higher_dimensions(self, data, target_dim):
        if self.default_handling == "auto" or self.default_handling == "pca":
            print(f"Reducing features using PCA to match {target_dim} dimensions...")
            reducer = DimensionalityReducer(target_dim)
            data.x = reducer.fit_transform(data.x)
        else:
            raise ValueError(f"Unsupported handling method for higher dimensions: {self.default_handling}")

    def compute_metrics(self, true, pred):
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average="macro")
        recall = recall_score(true, pred, average="macro")
        f1 = f1_score(true, pred, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")


if __name__ == "__main__":
    evaluator = CrossDatasetEvaluator(Config, default_handling="auto")
    evaluator.evaluate()
