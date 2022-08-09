import numpy as np


class CategoricalDistribution:
    def __init__(self, categories, initial_params=None):
        self.categories = categories
        # Uniform if none
        if initial_params is None:
            self.theta = np.ones(shape=(len(self.categories),))
            self.theta = self.theta * (1.0 / self.theta.shape[0])
        else:
            self.theta = initial_params

    def sample(self, num_of_samples):
        sampled = np.random.choice(a=self.categories, size=num_of_samples, p=self.theta)
        return sampled

    def maximum_likelihood_estimate(self, data, alpha):
        # We assume that data is a N x len(self.categories) array, with each row containing a one-hot vector.
        # If i. entry is 1, then it means that row has selected self.categories[i]
        categories_np = np.expand_dims(np.array(self.categories), axis=0)
        data_np = np.expand_dims(np.array(data), axis=1)
        categories_comparison_arr = (data_np == categories_np).astype(dtype=np.int32)
        assert np.array_equal(np.sum(categories_comparison_arr, axis=1).astype(dtype=np.int32),
                              np.ones(shape=(len(data),), dtype=np.int32))
        N_ = np.sum(categories_comparison_arr, axis=0)
        new_theta = N_ / categories_comparison_arr.shape[0]
        assert np.allclose(np.sum(new_theta), 1.0)
        self.theta = alpha * new_theta + (1.0 - alpha) * self.theta
