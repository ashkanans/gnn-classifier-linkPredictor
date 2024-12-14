class Config:
    EPOCHS = 200
    LEARNING_RATE = 0.01
    HIDDEN_DIM = 16
    WEIGHT_DECAY = 5e-4
    MODEL_SAVE_PATH = "models/gnn_model.pth"

    # Placeholder for input and output dimensions (Cora dataset values by default)
    INPUT_DIM = 1433
    OUTPUT_DIM = 7
