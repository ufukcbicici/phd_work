import numpy as np


# Not an actual distribution
class ConstantDistribution:
    def __init__(self, value, initial_params=None):
        self.value = value

    def sample(self, num_of_samples):
        sampled = np.zeros(shape=(num_of_samples, ))
        sampled[:] = self.value
        return sampled

    def maximum_likelihood_estimate(self, data, alpha):
        # We assume that data is a N x len(self.categories) array, with each row containing a one-hot vector.
        # If i. entry is 1, then it means that row has selected self.categories[i]
        pass
