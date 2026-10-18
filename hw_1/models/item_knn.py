import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, interactions):
        self.x = interactions.astype(np.float64).tocsr()
        gram = (self.x.T @ self.x).toarray()
        norms = np.sqrt(np.diag(gram))
        sim = gram / np.maximum(np.outer(norms, norms) + self.shrinkage, 1e-12)
        np.fill_diagonal(sim, 0)


        k = min(self.neighbors, sim.shape[0] - 1)
        for item in range(sim.shape[1]):
            order = np.argsort(-sim[:, item], kind="stable")
            sim[order[k:], item] = 0

        self.similarity = sparse.csc_matrix(sim)

        return self

    def score(self, users):
        return (self.x[users] @ self.similarity).toarray()
