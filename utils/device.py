import torch


class DeviceHandler:
    @staticmethod
    def get_device():
        """Returns the available device (CUDA if available, else CPU)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def move_model_to_device(model):
        """Moves the given model to the appropriate device."""
        device = DeviceHandler.get_device()
        model.to(device)
        return model, device

    @staticmethod
    def move_data_to_device(data, device):
        """Moves the data tensors to the specified device."""
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
        data.train_mask = data.train_mask.to(device)
        data.test_mask = data.test_mask.to(device)
        return data
