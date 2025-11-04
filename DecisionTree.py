import numpy as np


class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = 100000000000
        num_features = X.shape[1]
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left)*self.gini(left) +
                        len(right)*self.gini(right)) / len(y)
                if gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = gini
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth:
            return np.bincount(y).argmax()
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left_branch = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_branch = self.build_tree(X[right_idx], y[right_idx], depth+1)
        return (feature, threshold, left_branch, right_branch)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, node):
        if isinstance(node, int):
            return node
        feat, thresh, left, right = node
        return self.predict_one(x, left if x[feat] <= thresh else right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
