import torch
from sklearn.decomposition import PCA


def zero_pad_features(features, target_dim):
    """
    Zero-pad the features to match the target dimension.
    """
    current_dim = features.shape[1]
    padding = target_dim - current_dim
    return torch.nn.functional.pad(features, (0, padding), "constant", 0)


def replicate_features(features, target_dim):
    """
    Replicate the features to match the target dimension.
    """
    current_dim = features.shape[1]
    repeats = (target_dim + current_dim - 1) // current_dim
    return features.repeat(1, repeats)[:, :target_dim]


class DimensionalityReducer:
    def __init__(self, target_dim):
        self.target_dim = target_dim
        self.pca = PCA(n_components=target_dim)

    def fit_transform(self, features):
        """
        Fit PCA on the features and transform them.
        """
        features_np = features.cpu().numpy()
        reduced_features = self.pca.fit_transform(features_np)
        return torch.tensor(reduced_features, dtype=torch.float)
