import numpy as np
import matplotlib.pyplot as plt
import math

class RegressionTree:
    def __init__(self, max_depth=20, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(X) < self.min_samples_leaf:
            return np.mean(y)

        best_feature, best_split = self._get_best_split(X, y)

        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] < best_split
        right_indices = X[:, best_feature] >= best_split

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth+1)

        return (best_feature, best_split, left_tree, right_tree)

    def _get_best_split(self, X, y):
        best_score = np.inf
        best_feature = None
        best_split = None

        for feature in range(X.shape[1]):
            for split in np.unique(X[:, feature]):
                left_indices = X[:, feature] < split
                right_indices = X[:, feature] >= split

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_mean = np.mean(y[left_indices])
                right_mean = np.mean(y[right_indices])
                # score = np.sum((y[left_indices] - left_mean) ** 2) + np.sum((y[right_indices] - right_mean) ** 2)


                left_entropy = 0 if left_mean == 0 else -left_mean * math.log2(left_mean)
                right_entropy = 0 if right_mean == 0 else -right_mean * math.log2(right_mean)

                score = left_entropy + right_entropy


                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_split = split

        return best_feature, best_split

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, float):
            return node

        feature, split, left_tree, right_tree = node

        if x[feature] < split:
            return self._traverse_tree(x, left_tree)
        else:
            return self._traverse_tree(x, right_tree)
