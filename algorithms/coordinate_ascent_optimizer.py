import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class CoordinateAscentOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def maximizer(bounds, p0, sample_count_per_coordinate, func, max_iter=1000, tol=1e-5):
        iter_count = 0
        curr_coord_idx = 0
        dim = len(bounds)
        p = np.copy(p0)
        curr_max_y = func(p)
        y_list = [curr_max_y]
        while True:
            x = np.repeat(np.expand_dims(p, axis=0), repeats=sample_count_per_coordinate, axis=0)
            min_x, max_x = bounds[curr_coord_idx]
            ix = np.linspace(min_x, max_x, endpoint=True, num=sample_count_per_coordinate)
            x[:, curr_coord_idx] = ix
            y = func(x)
            y_max = np.max(y)
            y_argmax = np.argmax(y)
            p_max = x[y_argmax, :]
            iter_count += 1
            if iter_count == max_iter:
                break
            if np.abs(curr_max_y - y_max) < tol:
                break
            curr_coord_idx = (curr_coord_idx + 1) % dim
            p = p_max
            curr_max_y = y_max
            y_list.append(curr_max_y)
            print("X")


def experiment():
    mean0 = np.array([0.0, 0.0])
    cov0 = np.array([[0.1, 0.02], [0.02, 0.2]])
    rv0 = multivariate_normal(mean=mean0, cov=cov0)

    mean1 = np.array([0.8, 0.8])
    cov1 = np.array([[0.2, -0.05], [-0.05, 0.1]])
    rv1 = multivariate_normal(mean=mean1, cov=cov1)

    mean2 = np.array([-0.8, -0.8])
    cov2 = np.array([[0.15, -0.08], [-0.08, 0.15]])
    rv2 = multivariate_normal(mean=mean2, cov=cov2)

    func = lambda x: (1.0 / 3.0) * rv0.pdf(x) + (1.0 / 3.0) * rv1.pdf(x) + (1.0 / 3.0) * rv2.pdf(x)
    p0 = np.array([-1.6, 1.7])
    bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    CoordinateAscentOptimizer.maximizer(bounds=bounds, p0=p0, func=func, sample_count_per_coordinate=10000,
                                        max_iter=1000)

    # _x = np.linspace(-2, 2, 1000)
    # _y = np.linspace(-2, 2, 1000)
    # x_ax, y_ax = np.meshgrid(_x, _y)
    # coords = np.stack([x_ax, y_ax], axis=2)
    # z = func(coords)
    # plt.contourf(x_ax, y_ax, z, levels=100)
    # plt.show()
    print("X")


experiment()
