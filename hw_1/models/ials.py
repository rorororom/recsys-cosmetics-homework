import numpy as np


class ImplicitALS:
    def __init__(self, factors=24, regularization=1., alpha=10., iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, counts, preferences):

        extra = counts.copy()
        extra.data = self.alpha * np.log1p(extra.data)
        rhs_weights = preferences.copy()
        rhs_weights.data *= 1. + extra.data

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .01, (counts.shape[0], self.factors))
        self.item_factors = rng.normal(0, .01, (counts.shape[1], self.factors))

        for _ in range(self.iterations):
            self.user_factors = self._solve(extra, rhs_weights, self.item_factors)
            self.item_factors = self._solve(extra.T.tocsr(), rhs_weights.T.tocsr(), self.user_factors)

        return self

    def _solve(self, extra, rhs_weights, fixed):
        base = np.dot(fixed.T, fixed) + self.regularization * np.eye(self.factors)
        result = np.zeros((extra.shape[0], self.factors))

        for row in range(extra.shape[0]):
            start, end = extra.indptr[row:row + 2]
            indices = extra.indices[start:end]
            vectors = fixed[indices]
            confidence = extra.data[start:end]
            target = rhs_weights.data[start:end]

            a = base + np.dot(vectors.T * confidence, vectors)
            b = np.dot(vectors.T, target)
            result[row] = np.linalg.solve(a, b)

        if not np.isfinite(result).all():
            raise FloatingPointError("ALS produced non-finite factors")

        return result

    def score(self, users):
        return np.dot(self.user_factors[users], self.item_factors.T)
