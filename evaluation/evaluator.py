import torch
from models.gnn_model import GNNModel
from data.dataset_loader import DatasetLoader
from utils.config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class GNNEvaluator:
    def __init__(self, config):
        self.config = config
        self.dataset = DatasetLoader().load()
        self.data = self.dataset[0]
        self.model = GNNModel(
            input_dim=self.dataset.num_features,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.dataset.num_classes,
        )
        self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH))
        self.model.eval()

    def evaluate(self):
        with torch.no_grad():
            logits = self.model(self.data)
            pred = logits[self.data.test_mask].argmax(dim=1)
            true = self.data.y[self.data.test_mask]

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
    evaluator = GNNEvaluator(Config)
    evaluator.evaluate()
