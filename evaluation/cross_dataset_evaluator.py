import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import DatasetLoader
from models.gnn_model import GNNModel
from utils.config import Config


class CrossDatasetEvaluator:
    def __init__(self, config, datasets=["CiteSeer", "PubMed"]):
        self.config = config
        self.datasets = datasets
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

            with torch.no_grad():
                logits = self.model(data)
                pred = logits[data.test_mask].argmax(dim=1)
                true = data.y[data.test_mask]

            self.compute_metrics(true, pred)

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
    evaluator = CrossDatasetEvaluator(Config)
    evaluator.evaluate()
