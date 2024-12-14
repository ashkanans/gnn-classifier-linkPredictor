import torch
import torch.optim as optim

from data.dataset_loader import DatasetLoader
from models.generalized_gnn import GeneralizedGNN
from models.gnn_model import GNNModel
from utils.config import Config


def train_simple_gnn_model():
    # Load the dataset
    dataset = DatasetLoader("Cora").load()
    data = dataset[0]

    # Initialize the simple GNN model
    model = GNNModel(input_dim=dataset.num_features, hidden_dim=16, output_dim=dataset.num_classes)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Training loop
    model.train()
    for epoch in range(Config.EPOCHS):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Simple GNN model saved to", Config.MODEL_SAVE_PATH)


def train_generalized_gnn_model(variant, num_layers, hidden_dim, dropout, use_residual, use_layer_norm):
    # Load the dataset
    dataset = DatasetLoader("Cora").load()
    data = dataset[0]

    # Initialize the generalized GNN model
    model = GeneralizedGNN(
        input_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        num_layers=num_layers,
        variant=variant,
        dropout=dropout,
        use_residual=use_residual,
        use_layer_norm=use_layer_norm,
    )

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Training loop
    model.train()
    for epoch in range(Config.EPOCHS):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("Generalized GNN model saved to", Config.MODEL_SAVE_PATH)
