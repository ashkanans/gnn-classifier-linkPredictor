import torch
import torch.optim as optim

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.config import Config
from utils.device import DeviceHandler


class GNNTrainer:
    def __init__(self, model_type, hidden_dim, num_layers=2, variant="gcn", dropout=0.5, use_residual=False,
                 use_layer_norm=False):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.variant = variant
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Load the dataset
        self.dataset = DatasetLoader("Cora").load()
        self.data = self.dataset[0]

        # Move data to the device
        self.device = DeviceHandler.get_device()
        self.data = DeviceHandler.move_data_to_device(self.data, self.device)

        # Initialize the appropriate model
        self.model = self._initialize_model()
        self.model, self.device = DeviceHandler.move_model_to_device(self.model)

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    def _initialize_model(self):
        """Initializes and returns the appropriate model based on the model type."""
        if self.model_type == "simple":
            return GNNModel(input_dim=self.dataset.num_features, hidden_dim=self.hidden_dim,
                            output_dim=self.dataset.num_classes)
        elif self.model_type == "generalized":
            return GeneralizedGNN(
                input_dim=self.dataset.num_features,
                hidden_dim=self.hidden_dim,
                output_dim=self.dataset.num_classes,
                num_layers=self.num_layers,
                variant=self.variant,
                dropout=self.dropout,
                use_residual=self.use_residual,
                use_layer_norm=self.use_layer_norm
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self):
        """Trains the model."""
        print(f"\nStarting training for {self.model_type.capitalize()} GNN model...")
        self.model.train()
        for epoch in range(Config.EPOCHS):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = torch.nn.functional.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss.item():.4f}")

        # Save the trained model
        torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"{self.model_type.capitalize()} GNN model saved to", Config.MODEL_SAVE_PATH)
