import numpy as np


class EASE:
    def __init__(self, regularization=100.):
        self.regularization = regularization

    def fit(self, interactions):
        self.x = interactions.astype(np.float64).tocsr()
        gram = (self.x.T @ self.x).toarray()
        identity = np.eye(gram.shape[0])
        gram += self.regularization * identity

        precision = np.linalg.solve(gram, identity)
        self.coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.coefficients, 0.)

        return self

    def score(self, users):
        return np.asarray(self.x[users] @ self.coefficients)
