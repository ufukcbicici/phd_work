import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class CoordinateAscentOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def maximizer(bounds, p0, sample_count_per_coordinate, func, max_iter=1000, tol=1e-20):
        dim = len(bounds)
        curr_p = np.copy(p0)
        curr_max_y = func(curr_p)
        y_list = [curr_max_y]
        p_list = [curr_p]
        if dim == 2:
            CoordinateAscentOptimizer.draw_figure(
                min_x=bounds[0][0], max_x=bounds[0][1], min_y=bounds[1][0], max_y=bounds[1][1],
                points=p_list, func=func)
        for iter_id in range(max_iter):
            iter_max_y = curr_max_y
            iter_max_p = curr_p
            for curr_coord_idx in range(dim):
                x = np.repeat(np.expand_dims(iter_max_p, axis=0), repeats=sample_count_per_coordinate, axis=0)
                min_x, max_x = bounds[curr_coord_idx]
                ix = np.linspace(min_x, max_x, endpoint=True, num=sample_count_per_coordinate)
                x[:, curr_coord_idx] = ix
                y = func(x)
                y_max = np.max(y)
                y_argmax = np.argmax(y)
                p_max = x[y_argmax, :]
                if y_max > iter_max_y + tol:
                    iter_max_y = y_max
                    iter_max_p = p_max
                    y_list.append(iter_max_y)
                    p_list.append(iter_max_p)
                    if dim == 2:
                        CoordinateAscentOptimizer.draw_figure(
                            min_x=bounds[0][0], max_x=bounds[0][1], min_y=bounds[1][0], max_y=bounds[1][1],
                            points=p_list, func=func)
            if iter_max_y > curr_max_y + tol:
                curr_max_y = iter_max_y
                curr_p = iter_max_p
            else:
                break
        return curr_max_y, curr_p

    @staticmethod
    def draw_figure(min_x, max_x, min_y, max_y, points, func):
        _x = np.linspace(min_x, max_x, 1000)
        _y = np.linspace(min_y, max_y, 1000)
        x_ax, y_ax = np.meshgrid(_x, _y)
        coords = np.stack([x_ax, y_ax], axis=2)
        z = func(coords)
        fig, ax = plt.subplots()
        ax.contourf(x_ax, y_ax, z, levels=100)
        points_stacked = np.stack(points, axis=0)
        ax.plot(points_stacked[:, 0], points_stacked[:, 1], 'r+', ms=3)
        fig.savefig('coord_ascent_{0}.png'.format(points_stacked.shape[0]))


def experiment():
    mean0 = np.array([0.0, 0.0])
    cov0 = np.array([[0.1, 0.02], [0.02, 0.2]])
    rv0 = multivariate_normal(mean=mean0, cov=cov0)

    mean1 = np.array([0.8, 0.8])
    cov1 = np.array([[0.2, -0.05], [-0.05, 0.1]])
    rv1 = multivariate_normal(mean=mean1, cov=cov1)

    mean2 = np.array([-0.8, -0.8])
    cov2 = np.array([[0.05, -0.08], [-0.08, 0.15]])
    rv2 = multivariate_normal(mean=mean2, cov=cov2)

    func = lambda x: (1.0 / 3.0) * rv0.pdf(x) + (1.0 / 3.0) * rv1.pdf(x) + (1.0 / 3.0) * rv2.pdf(x)
    p0 = np.random.uniform(-2.0, 2.0, size=(2, ))
    bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    CoordinateAscentOptimizer.maximizer(bounds=bounds, p0=p0, func=func, sample_count_per_coordinate=10000,
                                        max_iter=1000)

    # _x = np.linspace(-2, 2, 1000)
    # _y = np.linspace(-2, 2, 1000)
    # x_ax, y_ax = np.meshgrid(_x, _y)
    # coords = np.stack([x_ax, y_ax], axis=2)
    # z = func(coords)
    # fig, ax = plt.subplots()
    # # ax = plt.contourf(x_ax, y_ax, z, levels=100)
    # ax.contourf(x_ax, y_ax, z, levels=100)
    # x = np.array([0.5])
    # y = np.array([0.5])
    # ax.plot(x, y, 'ko', ms=3)
    # fig.savefig('ax1_figure.png')
    # x = np.array([0.6])
    # y = np.array([0.6])
    # ax.plot(x, y, 'ko', ms=3)
    # fig.savefig('ax2_figure.png')
    # plt.show()
    # print("X")


experiment()
