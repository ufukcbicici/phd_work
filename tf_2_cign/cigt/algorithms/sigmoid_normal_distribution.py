import numpy as np
from scipy.stats import norm


class SigmoidNormalDistribution:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_of_samples):
        # Sample from normal first
        s = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_of_samples, ))
        s2 = 1.0 + np.exp(-s)
        y = np.reciprocal(s2)
        return y

    def pdf(self, y):
        assert 0.0 <= y <= 1.0
        x = np.log(y / (1.0 - y))

        # sampled = np.random.choice(a=self.categories, size=num_of_samples, p=self.theta)
        # return sampled

    # def maximum_likelihood_estimate(self, data, alpha):
    #     # We assume that data is a N x len(self.categories) array, with each row containing a one-hot vector.
    #     # If i. entry is 1, then it means that row has selected self.categories[i]
    #     N_ = np.sum(data, axis=0)
    #     new_theta = N_ / data.shape[0]
    #     assert np.allclose(np.sum(new_theta), 1.0)
    #     self.theta = alpha * new_theta + (1.0 - alpha) * self.theta
