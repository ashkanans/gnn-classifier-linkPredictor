import json
import os
from datetime import datetime

import torch


class ModelSaver:
    @staticmethod
    def create_save_directory(base_path, method_type):
        """Creates a directory based on method_type and current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_path, f"{method_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    @staticmethod
    def save_model(model, save_dir, model_filename="model.pth"):
        """Saves the model's state dictionary."""
        model_path = os.path.join(save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def save_metadata(save_dir, metadata, metadata_filename="metadata.json"):
        """Saves the metadata as a JSON file."""
        metadata_path = os.path.join(save_dir, metadata_filename)

        # Add GPU info at the top of metadata
        gpu_info = ModelSaver.get_gpu_info()
        metadata["gpu_info"] = gpu_info

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")

    @staticmethod
    def update_metadata(save_dir, updates, metadata_filename="metadata.json"):
        """Updates the metadata JSON file with new information."""
        metadata_path = os.path.join(save_dir, metadata_filename)

        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update metadata with new information
        metadata.update(updates)

        # Save the updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata updated with {list(updates.keys())} and saved to {metadata_path}")

    @staticmethod
    def get_gpu_info():
        """Returns GPU information if CUDA is available, otherwise None."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return {"gpu_used": True, "gpu_name": gpu_name}
        else:
            return {"gpu_used": False, "gpu_name": None}
