import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None  # Will store the tree structure

    # ------------------------
    # Gini Impurity
    # ------------------------
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    # ------------------------
    # Split the dataset
    # ------------------------
    def split_dataset(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    # ------------------------
    # Find best split
    # ------------------------
    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature, t)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                g_left = self.gini(y_left)
                g_right = self.gini(y_right)
                g = (len(y_left)/n_samples) * g_left + (len(y_right)/n_samples) * g_right

                if g < best_gini:
                    best_gini = g
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    # ------------------------
    # Build tree recursively
    # ------------------------
    def build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_labels = len(np.unique(y))

        # stopping conditions
        if depth >= self.max_depth or num_samples < self.min_samples_split or num_labels == 1:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": leaf_value}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": leaf_value}

        X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature, threshold)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X_left, y_left, depth + 1),
            "right": self.build_tree(X_right, y_right, depth + 1)
        }

    # ------------------------
    # Train
    # ------------------------
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # ------------------------
    # Predict single sample
    # ------------------------
    def predict_one(self, x, node):
        if "leaf" in node:
            return node["leaf"]
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    # ------------------------
    # Predict array
    # ------------------------
    def predict(self, X):
        return np.array([self.predict_one(sample, self.tree) for sample in X])
