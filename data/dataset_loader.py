from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms import RandomLinkSplit


class DatasetLoader:
    def __init__(self, dataset_name, root_dir="data/datasets"):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load(self):
        """
        Load the dataset and apply normalization.
        """
        dataset = Planetoid(
            root=f"{self.root_dir}/{self.dataset_name}",
            name=self.dataset_name,
            transform=NormalizeFeatures(),
        )
        self.data = dataset[0]
        return dataset

    def prepare_edges(self):
        """
        Splits the edges of the graph into train/validation/test sets for link prediction using RandomLinkSplit.
        """
        if self.data is None:
            raise RuntimeError("Dataset must be loaded before preparing edges.")

        # Use RandomLinkSplit for edge splitting
        transform = RandomLinkSplit(
            is_undirected=True,  # Assume undirected graph
            num_val=0.1,  # 10% of edges for validation
            num_test=0.2,  # 20% of edges for testing
        )
        self.train_data, self.val_data, self.test_data = transform(self.data)

    def get_label_distribution(self, labels):
        """
        Computes the distribution of positive and negative labels.
        """
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 0).sum().item()
        return pos_count, neg_count

    def get_train_edges(self):
        """
        Returns positive and negative edges for training.
        """
        if self.train_data is None:
            raise RuntimeError("Edges must be prepared before accessing train edges.")

        return self.train_data.edge_label_index, self.train_data.edge_label

    def get_val_edges(self):
        """
        Returns positive and negative edges for validation.
        """
        if self.val_data is None:
            raise RuntimeError("Edges must be prepared before accessing validation edges.")

        return self.val_data.edge_label_index, self.val_data.edge_label

    def get_test_edges(self):
        """
        Returns positive and negative edges for testing.
        """
        if self.test_data is None:
            raise RuntimeError("Edges must be prepared before accessing test edges.")

        return self.test_data.edge_label_index, self.test_data.edge_label


if __name__ == "__main__":
    # Test loading and preparing datasets for link prediction
    for name in ["Cora", "CiteSeer", "PubMed"]:
        loader = DatasetLoader(dataset_name=name)
        dataset = loader.load()
        print(f"{name} Dataset:")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print("-" * 30)

        # Prepare edges for link prediction
        loader.prepare_edges()
        train_edges, train_labels = loader.get_train_edges()
        val_edges, val_labels = loader.get_val_edges()
        test_edges, test_labels = loader.get_test_edges()

        # Print edge counts and label distributions
        train_pos, train_neg = loader.get_label_distribution(train_labels)
        val_pos, val_neg = loader.get_label_distribution(val_labels)
        test_pos, test_neg = loader.get_label_distribution(test_labels)

        print(f"Train Edges: {train_edges.shape[1]}")
        print(f"Train Labels: Positive: {train_pos}, Negative: {train_neg}")
        print(f"Validation Edges: {val_edges.shape[1]}")
        print(f"Validation Labels: Positive: {val_pos}, Negative: {val_neg}")
        print(f"Test Edges: {test_edges.shape[1]}")
        print(f"Test Labels: Positive: {test_pos}, Negative: {test_neg}")
        print("=" * 50)
