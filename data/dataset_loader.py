from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class DatasetLoader:
    def __init__(self, dataset_name, root_dir="data/datasets"):
        self.dataset_name = dataset_name
        self.root_dir = root_dir

    def load(self):
        dataset = Planetoid(root=f"{self.root_dir}/{self.dataset_name}", name=self.dataset_name, transform=NormalizeFeatures())
        return dataset


if __name__ == "__main__":
    # Test loading different datasets
    for name in ["Cora", "CiteSeer", "PubMed"]:
        loader = DatasetLoader(dataset_name=name)
        dataset = loader.load()
        print(f"{name} Dataset:")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print("-" * 30)
