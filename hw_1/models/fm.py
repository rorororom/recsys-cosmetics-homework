import numpy as np
from scipy.special import expit


class FactorizationMachine:
    def __init__(self, factors=16, learning_rate=.04, regularization=1e-4,
                 epochs=6, batch_size=2048, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    @staticmethod
    def logits(features, w, v, bias):
        embeddings = v[features]
        summed = embeddings.sum(axis=1)
        linear = bias + w[features].sum(axis=1)
        interactions = .5 * (summed**2 - (embeddings**2).sum(axis=1)).sum(axis=1)

        return linear + interactions

    def fit(self, users, items, labels, item_features, n_users):
        self.item_features = np.asarray(item_features, dtype=np.int64)
        features = np.column_stack([users, self.item_features[items]])
        n_features = max(n_users, int(self.item_features.max()) + 1)

        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(n_features)
        self.v = rng.normal(0, .05, (n_features, self.factors))
        self.bias = 0.

        gw_acc = np.full_like(self.w, 1e-6)
        gv_acc = np.full_like(self.v, 1e-6)
        gb_acc = 1e-6
        n_batches = max(1, int(np.ceil(len(labels) / self.batch_size)))

        for _ in range(self.epochs):
            order = rng.permutation(len(labels))

            for batch in np.array_split(order, n_batches):
                f = features[batch]
                y = labels[batch]
                emb = self.v[f]
                summed = emb.sum(axis=1)
                pred = self.logits(f, self.w, self.v, self.bias)
                err = (expit(pred) - y) / len(batch)


                unique, inverse = np.unique(f, return_inverse=True)
                inverse = inverse.reshape(f.shape)
                dw = np.zeros(len(unique))
                dv = np.zeros((len(unique), self.factors))
                for column in range(f.shape[1]):
                    indices = inverse[:, column]
                    gradient = err[:, None] * (summed - emb[:, column])
                    np.add.at(dw, indices, err)
                    np.add.at(dv, indices, gradient)


                freq = np.bincount(inverse.ravel(), minlength=len(unique)) / len(batch)
                dw += self.regularization * freq * self.w[unique]
                dv += self.regularization * freq[:, None] * self.v[unique]


                gw_acc[unique] += dw**2
                gv_acc[unique] += dv**2
                self.w[unique] -= self.learning_rate * dw / np.sqrt(gw_acc[unique])
                self.v[unique] -= self.learning_rate * dv / np.sqrt(gv_acc[unique])

                db = err.sum()
                gb_acc += db**2
                self.bias -= self.learning_rate * db / np.sqrt(gb_acc)

        if not (np.isfinite(self.w).all() and np.isfinite(self.v).all() and np.isfinite(self.bias)):
            raise FloatingPointError("FM produced non-finite parameters")


        emb = self.v[self.item_features]
        self.item_sum = emb.sum(axis=1)
        self.item_bias = (self.w[self.item_features].sum(axis=1)
                          + .5 * (self.item_sum**2 - (emb**2).sum(axis=1)).sum(axis=1))
        return self

    def score(self, users):
        users = np.asarray(users)
        return (self.bias + self.w[users, None] + self.item_bias[None, :]
                + np.dot(self.v[users], self.item_sum.T))
