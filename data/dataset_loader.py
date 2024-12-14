import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class DatasetLoader:
    def __init__(self, dataset_name="Cora", root_dir="data/datasets"):
        self.dataset_name = dataset_name
        self.root_dir = root_dir

    def load(self):
        dataset = Planetoid(root=f"{self.root_dir}/{self.dataset_name}", name=self.dataset_name, transform=NormalizeFeatures())
        return dataset


if __name__ == "__main__":
    loader = DatasetLoader()
    dataset = loader.load()
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
