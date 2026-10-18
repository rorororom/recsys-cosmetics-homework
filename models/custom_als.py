import numpy as np
from .ials import ImplicitALS


class CustomTargetALS(ImplicitALS):
    def fit(self, counts, event_matrices, age_days):
        weights = {'view': 1., 'cart': 2., 'purchase': 4., 'remove_from_cart': -2.}
        signal = np.zeros_like(counts.data)

        for event, weight in weights.items():
            signal += weight * np.log1p(event_matrices[event].data)

        target = counts.copy()
        target.data = np.tanh(signal / 3.) * np.exp2(-age_days / 14.)

        return super().fit(counts, target)
