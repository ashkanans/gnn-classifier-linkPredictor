import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(embeddings, method="pca"):
    """
    Reduces dimensions of embeddings to 2D.

    Args:
        embeddings (torch.Tensor): High-dimensional embeddings.
        method (str): Method for dimensionality reduction ('pca' or 'tsne').

    Returns:
        numpy.ndarray: 2D reduced embeddings.
    """
    if method == "pca":
        return PCA(n_components=2).fit_transform(embeddings)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError("Unsupported reduction method. Use 'pca' or 'tsne'.")


def visualize_embeddings(embeddings, labels, method="pca", output_path=None):
    """
    Visualizes embeddings in 2D space.

    Args:
        embeddings (torch.Tensor): High-dimensional embeddings.
        labels (torch.Tensor): Node labels for color coding.
        method (str): Method for dimensionality reduction ('pca' or 'tsne').
        output_path (str): Optional path to save the plot.
    """
    reduced = reduce_dimensions(embeddings, method=method)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Node Classes")
    plt.title(f"Node Embeddings Visualized using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show(block=True)
