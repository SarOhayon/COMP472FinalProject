import numpy as np


class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feat, best_thresh, best_gini = None, None, 1.0
        current_gini = self.gini(y)

        for feat in range(n_features):
            sorted_idx = np.argsort(X[:, feat])
            Xf, yf = X[sorted_idx, feat], y[sorted_idx]

            num_left = np.zeros(10)
            num_right = np.bincount(yf, minlength=10).astype(float)
            total_samples = len(yf)

            for i in range(1, total_samples):
                cls = yf[i - 1]
                num_left[cls] += 1
                num_right[cls] -= 1

                if Xf[i] == Xf[i - 1]:
                    continue

                left_size = i
                right_size = total_samples - i

                g_left = 1.0 - np.sum((num_left[:10] / left_size) ** 2)
                g_right = 1.0 - np.sum((num_right[:10] / right_size) ** 2)
                g = (left_size * g_left + right_size * g_right) / total_samples

                if g < best_gini:
                    best_gini = g
                    best_feat = feat
                    best_thresh = (Xf[i] + Xf[i - 1]) / 2.0

        if best_feat is None or best_gini >= current_gini:
            return None, None

        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return np.bincount(y).argmax()

        left_idx = X[:, feat] <= thresh
        right_idx = X[:, feat] > thresh

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return feat, thresh, left, right

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, node):
        if isinstance(node, (int, np.integer)):
            return node
        feat, thresh, left, right = node
        if x[feat] <= thresh:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
