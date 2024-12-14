import torch
import torch.optim as optim
from models.gnn_model import GNNModel
from data.dataset_loader import DatasetLoader
from utils.config import Config


class GNNTrainer:
    def __init__(self, config):
        self.config = config
        self.dataset = DatasetLoader().load()
        self.data = self.dataset[0]
        self.model = GNNModel(
            input_dim=self.dataset.num_features,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.dataset.num_classes,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY
        )

    def train(self):
        self.model.train()
        for epoch in range(self.config.EPOCHS):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = torch.nn.functional.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}, Loss: {loss.item():.4f}")

        self.save_model()

    def save_model(self):
        model_path = self.config.MODEL_SAVE_PATH
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    from utils.config import Config

    trainer = GNNTrainer(Config)
    trainer.train()
