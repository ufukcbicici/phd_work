import numpy as np
import os
from scipy.stats import norm
import tensorflow as tf
from tqdm import tqdm
from scipy import integrate
import matplotlib.pyplot as plt


class SigmoidNormalDistribution:
    def __init__(self, low_end=0.0, high_end=1.0, name="", mu=0.0, sigma=1.0):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.lowEnd = low_end
        self.highEnd = high_end

    def sample(self, num_of_samples):
        # Sample from normal first
        s = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_of_samples, ))
        s2 = 1.0 + np.exp(-s)
        y = np.reciprocal(s2)
        # Convert into target interval
        a = self.lowEnd
        b = self.highEnd
        y_hat = (b - a)*y + a
        return y_hat

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

    def maximum_likelihood_estimate(self, data, alpha):
        # Convert data to the original scale
        a = self.lowEnd
        b = self.highEnd
        data = (data - a) * (1.0 / (b - a))
        data = np.clip(a=data, a_min=0.0 + 1.0e-10, a_max=1.0 - 1.0e-10)
        y_hat = data / (1.0 - data)
        x = np.log(y_hat)
        mu_hat = np.mean(x)
        diff = np.square(x - mu_hat)
        sigma_hat_squared = np.mean(diff)
        sigma_hat = np.sqrt(sigma_hat_squared)
        if sigma_hat <= 0.0:
            sigma_hat = 1.0e-10
        self.mu = mu_hat
        self.sigma = sigma_hat

    def plot_distribution(self, root_path):
        samples = self.sample(num_of_samples=1000000)
        samples = (samples - self.lowEnd) / (self.highEnd - self.lowEnd)

        y = np.arange(0.0, 1.0, 0.0001)
        pdf_y = self.pdf(y=y)

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Distribution:{0} mu={1} sigma={2}".format(self.name, self.mu, self.sigma))
        ax.plot(y, pdf_y, 'k-', lw=2, label='pdf')
        ax.hist(samples, density=True, histtype='stepfilled', alpha=0.2, bins=100)
        ax.legend(loc='best', frameon=False)
        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(root_path, "{0}.png".format(self.name)), dpi=1000)
        plt.close()


def plot_and_integrate():
    sigmoid_normal = SigmoidNormalDistribution(mu=3.0, sigma=10.0)
    y = np.arange(1.e-30, 1.0 - 1.e-6, 0.0001)
    pdf_y = sigmoid_normal.pdf(y=y)

    plt.plot(y, pdf_y)
    plt.show()

    res = integrate.quadrature(sigmoid_normal.pdf, 0.0, 1.0, tol=0.0, maxiter=1000000)
    assert np.isclose(res[0], 1.0)


def plot_samples():
    sigmoid_normal = SigmoidNormalDistribution(mu=30.0, sigma=100.0)
    samples = sigmoid_normal.sample(num_of_samples=1000000)
    y = np.arange(0.0, 1.0, 0.0001)
    pdf_y = sigmoid_normal.pdf(y=y)

    fig, ax = plt.subplots(1, 1)
    ax.plot(y, pdf_y, 'k-', lw=2, label='pdf')
    ax.hist(samples, density=True, histtype='stepfilled', alpha=0.2, bins=100)
    ax.legend(loc='best', frameon=False)
    plt.show()
    print("X")


def check_mle():
    a_ = 1000.0
    b_ = 1000.0
    for it_ in range(1000):
        sigmoid_normal = SigmoidNormalDistribution(mu=a_, sigma=b_)
        samples = sigmoid_normal.sample(num_of_samples=1000000)
        y = np.arange(0.0, 1.0, 0.0001)
        pdf_y = sigmoid_normal.pdf(y=y)

        fig, ax = plt.subplots(1, 1)
        ax.plot(y, pdf_y, 'k-', lw=2, label='pdf')
        ax.hist(samples, density=True, histtype='stepfilled', alpha=0.2, bins=100)
        ax.legend(loc='best', frameon=False)
        plt.show()

        sigmoid_normal_mle = SigmoidNormalDistribution(mu=0.0, sigma=1.0)
        sigmoid_normal_mle.maximum_likelihood_estimate(data=samples, alpha=None)
        print("mu_mle={0} sigma_mle={1}".format(sigmoid_normal_mle.mu,
                                                sigmoid_normal_mle.sigma))
        a_ = sigmoid_normal_mle.mu
        b_ = sigmoid_normal_mle.sigma
        print("X")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # plot_samples()
    # check_mle()

    plot_and_integrate()
    #
    #
    #
    # plt.plot(y, pdf_y)
    # plt.show()
    #
    # res = integrate.quadrature(sigmoid_normal.pdf, 0.0, 1.0, tol=0.0, maxiter=1000000)
    # assert np.isclose(res[0], 1.0)
    #
    # print("X")

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
