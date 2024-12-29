import torch
import torch.optim as optim

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.config import Config
from utils.device import DeviceHandler
from utils.model_saver import ModelSaver


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

        # Load the dataset. "Cora" dataset is used for the initial training
        self.dataset = DatasetLoader("Cora").load()
        self.data = self.dataset[0]

        self.device = DeviceHandler.get_device()
        self.data = DeviceHandler.move_data_to_device(self.data, self.device)

        self.model = self._initialize_model()
        self.model, self.device = DeviceHandler.move_model_to_device(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

        self.save_dir = ModelSaver.create_save_directory("saved_models", self.model_type)
        self.last_loss = None

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
        """Trains the model and saves it with metadata."""
        print(f"\nStarting training for {self.model_type.capitalize()} GNN model...")
        self.model.train()
        for epoch in range(Config.EPOCHS):
            self.optimizer.zero_grad()
            out = self.model(data=self.data)
            loss = torch.nn.functional.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            self.last_loss = loss.item()
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss.item():.4f}")

        # Save the trained model
        ModelSaver.save_model(self.model, self.save_dir)

        # Save metadata
        metadata = self._generate_metadata()
        ModelSaver.save_metadata(self.save_dir, metadata)

    def _generate_metadata(self):
        """Generates metadata for the training session."""
        metadata = {
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "variant": self.variant,
            "dropout": self.dropout,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "dataset": {
                "name": "Cora",
                "num_features": self.dataset.num_features,
                "num_classes": self.dataset.num_classes,
                "num_nodes": self.data.x.shape[0]
            },
            "training": {
                "epochs": Config.EPOCHS,
                "learning_rate": Config.LEARNING_RATE,
                "weight_decay": Config.WEIGHT_DECAY,
                "last_epoch_loss": self.last_loss
            }
        }
        return metadata
