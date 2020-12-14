import numpy as np


class BootstrapMeanComparison:
    def __init__(self):
        pass

    @staticmethod
    def compare(x, y, boostrap_count, significance_level=0.05):
        x_mean = np.mean(x)
        x_var = np.var(x, ddof=1)
        n = x.shape[0]

        y_mean = np.mean(y)
        y_var = np.var(y, ddof=1)
        m = y.shape[0]

        t_A = (x_mean - y_mean)
        t_B = np.sqrt(x_var / n + y_var / m)
        t = t_A / t_B

        z = np.concatenate([x, y])
        z_mean = np.mean(z)
        x_hat = np.array([x_i - x_mean + z_mean for x_i in x])
        y_hat = np.array([y_i - y_mean + z_mean for y_i in y])

        # Bootstrap sampling
        comparison_arr = []
        for idx in range(boostrap_count):
            x_star = np.random.choice(x_hat, n, replace=True)
            y_star = np.random.choice(y_hat, m, replace=True)
            x_star_mean = np.mean(x_star)
            x_star_var = np.var(x_star, ddof=1)
            y_star_mean = np.mean(y_star)
            y_star_var = np.var(y_star, ddof=1)

            t_star_A = (x_star_mean - y_star_mean)
            t_star_B = np.sqrt(x_star_var / n + y_star_var / m)
            t_star = t_star_A / t_star_B
            if t > 0:
                comparison_arr.append(t_star >= t)
            else:
                comparison_arr.append(t_star <= t)
        p_value = np.sum(np.array(comparison_arr).astype(np.float32) / boostrap_count)
        reject_null_hypothesis = p_value <= significance_level
        return p_value, reject_null_hypothesis
