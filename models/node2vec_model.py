import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import Node2Vec


class Node2VecModel:
    def __init__(self, edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10, epochs=50):
        """
        Initialize the Node2Vec model.

        Args:
            edge_index (torch.Tensor): The edge index of the graph.
            embedding_dim (int): Dimensionality of the node embeddings.
            walk_length (int): Length of random walks.
            context_size (int): Context size for skip-gram.
            walks_per_node (int): Number of random walks per node.
            epochs (int): Number of training epochs.
        """
        self.edge_index = edge_index
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.epochs = epochs

        self.model = Node2Vec(
            edge_index=self.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=1
        )

    def train(self):
        """
        Train the Node2Vec model to generate embeddings.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Generate a batch of nodes (1D tensor of node indices)
            batch = torch.arange(self.model.num_nodes)

            # Repeat the batch for the number of walks per node
            batch = batch.repeat_interleave(self.walks_per_node)

            # Generate positive random walks
            pos_rw = self.model.pos_sample(batch)

            # Extract the first column of pos_rw to create a valid batch for neg_sample
            neg_batch = pos_rw[:, 0]

            # Generate negative random walks
            neg_rw = self.model.neg_sample(neg_batch)

            # Compute the loss
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        print("Training complete!")

    def get_embeddings(self):
        """
        Generate and return node embeddings.

        Returns:
            torch.Tensor: Node embeddings.
        """
        self.model.eval()
        return self.model.embedding.weight.data

    def evaluate_link_prediction(self, embeddings, pos_edge_index, neg_edge_index):
        """
        Evaluate link prediction using Node2Vec embeddings.
        """
        pos_scores = (embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))], dim=0)

        pred = scores > 0
        accuracy = accuracy_score(labels.numpy(), pred.numpy())
        precision = precision_score(labels.numpy(), pred.numpy())
        recall = recall_score(labels.numpy(), pred.numpy())
        f1 = f1_score(labels.numpy(), pred.numpy())

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    def evaluate_node_classification(self, embeddings, labels, train_mask, test_mask):
        """
        Evaluate node classification using Node2Vec embeddings.
        """
        X_train, X_test = embeddings[train_mask], embeddings[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test.numpy(), y_pred)
        precision = precision_score(y_test.numpy(), y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test.numpy(), y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test.numpy(), y_pred, average="macro", zero_division=0)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
