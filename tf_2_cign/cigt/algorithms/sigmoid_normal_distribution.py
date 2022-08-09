import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tqdm import tqdm
from scipy import integrate
import matplotlib.pyplot as plt


class SigmoidNormalDistribution:
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_of_samples):
        # Sample from normal first
        s = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_of_samples, ))
        s2 = 1.0 + np.exp(-s)
        y = np.reciprocal(s2)
        return y

    def pdf(self, y):
        assert np.all(np.greater_equal(y, 0.0))
        assert np.all(np.less_equal(y, 1.0))
        y_hat = y / (1.0 - y)
        x = np.log(y_hat)
        a = y - np.square(y)
        a = np.reciprocal(a)
        pdf_x = norm.pdf(x=x, loc=self.mu, scale=self.sigma)
        res = pdf_x * a
        return res


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    sigmoid_normal = SigmoidNormalDistribution(mu=1.5, sigma=0.1)
    y = np.arange(1.e-30, 1.0 - 1.e-6, 0.0001)
    pdf_y = sigmoid_normal.pdf(y=y)

    plt.plot(y, pdf_y)
    plt.show()

    print("X")

    # sample_count = 1000000
    # samples = []
    # for s_id in tqdm(range(sample_count)):
    #     y_ = np.random.uniform()
    #     y_pdf = sigmoid_normal.pdf(y=y_)
    #     samples.append(y_pdf)
    # area_estimated = np.mean(samples)
    # print("X")


        # sampled = np.random.choice(a=self.categories, size=num_of_samples, p=self.theta)
        # return sampled

    # def maximum_likelihood_estimate(self, data, alpha):
    #     # We assume that data is a N x len(self.categories) array, with each row containing a one-hot vector.
    #     # If i. entry is 1, then it means that row has selected self.categories[i]
    #     N_ = np.sum(data, axis=0)
    #     new_theta = N_ / data.shape[0]
    #     assert np.allclose(np.sum(new_theta), 1.0)
    #     self.theta = alpha * new_theta + (1.0 - alpha) * self.theta
