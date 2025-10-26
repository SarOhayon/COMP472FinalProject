from typing import Any
import numpy as np


class GaussianNaiveBayes:
    def __init__(self) -> None:
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

# X --> feature matrix(5000x512), y -->lables(vector of 5000 size)
    def fit(self, X, y):
        # grabbing all unique classes --> should yield 10
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        # grabbing second dimesion of 512 --> gives 512 features
        n_features = X.shape[1]
        # mean
        self.means = np.zeros((n_classes, n_features))
       # length of bell curve aka variances
        self.variances = np.zeros((n_classes, n_features))
        # probabilites of each class (will prbs be .1 always cuz only 10 classes and equal size training sets for each class)
        self.priors = np.zeros(n_classes)

        for index, cls in enumerate(self.classes):
            # slicing X, this will make x_c 500x512
            X_c = X[y == cls]
            self.means[index, :] = X_c.mean(axis=0)
            # adding small value close to zero so i never get error for diving by zero
            self.variances[index, :] = X_c.var(axis=0) + 1e-9
            self.priors[index] = X_c.shape[0]/X.shape[0]
