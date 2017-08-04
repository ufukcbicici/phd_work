import numpy as np
import itertools

# N = 20
# D = 784
# n = 4
# K = 5
# L = 100
# m = 25000


# def get_joint_distribution(data, h_choices):
#     p_xcn = np.zeros(shape=(N, L, K))
#     p_x = 1.0 / float(N)
#     p_xcn[:] = p_x
#     p_c_given_x = np.zeros(shape=(N, L, 1))
#     for x in range(N):
#         p_c_given_x[x, data[x]][:] = 1.0
#     p_xc = p_xcn * p_c_given_x
#     p_n_given_x = np.reshape(h_choices, newshape=(N, 1, K))
#     p_xcn = p_xc * p_n_given_x
#     return p_xcn
#     # for i in range(N):
#     #     for j in range(L):
#     #         for k in range(K):
#     #             p_x = 1.0 / float(N)
#     #             p_c_given_x = float(j == data[i])
#     #             p_n_given_x_c = h_choices[i, k]
#     #             p_xcn[i, j, k] = p_x * p_c_given_x * p_n_given_x_c
#     # return p_xcn


# def p_joint(x, c, n, data, h):
#     Np = data.shape[0]
#     p_x = 1.0 / float(Np)
#     p_c_given_x = float(c == data[x])
#     p_n_given_x_c = h[x, n]
#     p_x_c_n = p_x * p_c_given_x * p_n_given_x_c
#     return p_x_c_n
#
#
# def p_cn(c, n, data, h):
#     Np = data.shape[0]
#     p_c_n = 0.0
#     for i in range(Np):
#         p_c_n += p_joint(x=i, c=c, n=n, data=data, h=h)
#     return p_c_n
#
#
# def p_n(n, data, h):
#     Np = data.shape[0]
#     prob = 0.0
#     for i in range(Np):
#         for l in range(L):
#             prob += p_joint(x=i, c=l, n=n, data=data, h=h)
#     return prob


def generate_data(sample_count, decision_count, label_count, data_dim):
    N = sample_count
    K = decision_count
    L = label_count
    D = data_dim
    S = np.random.uniform(0.0, float(L), size=(N,)).astype(int)
    X = np.random.uniform(0.0, 1.0, size=(N, D))
    W = np.random.uniform(-1.0, 1.0, size=(K, D))
    b = np.random.uniform(-1.0, 1.0, size=(K, 1))
    H = np.transpose(np.dot(W, np.transpose(X)) + b)
    H = np.exp(H)
    # for i in range(N):
    #     alpha = np.random.uniform(5.0, 10.0)
    #     beta = np.random.uniform(5.0, 10.0)
    #     H[i][:] = np.random.beta(a=alpha, b=beta, size=(1, K))
    sums_h = H.sum(axis=1)
    h = H / sums_h.reshape(N, 1)
    return S, h, X


# def get_var(expectations, mean):
#     sub = expectations - mean
#     pow = np.power(sub, 2.0)
#     variance = pow.sum() / float(pow.shape[0])
#     return variance
#
#
# # Apply Rejection Sampling
# def rejection_sampling(samples, subset_of_target_distribution, max_num_of_samples):
#     target_distribution_samples = []
#     mode_dist = np.max(subset_of_target_distribution)
#     for i in range(len(samples)):
#         sample = samples[i]
#         a = np.random.uniform(low=0.0, high=mode_dist)
#         q_i = subset_of_target_distribution[i]
#         # Accept the sample
#         if a <= q_i:
#             target_distribution_samples.append((sample, q_i))
#         if len(target_distribution_samples) == max_num_of_samples:
#             break
#     return target_distribution_samples
#
#
# def estimate_log_prob(h):
#     estimated_prob = np.mean(h)
#     log_estimated_prob = np.log(estimated_prob)
#     sample_count = h.shape[0]
#     sample_variance = (np.power(h - estimated_prob, 2.0)).sum() / (float(sample_count) - 1.0)
#     taylor_correction = sample_variance / (2.0 * float(sample_count) * (estimated_prob ** 2))
#     corrected_log_estimate = log_estimated_prob + taylor_correction
#     return log_estimated_prob, corrected_log_estimate
#
#
# def generate_optimal_sampling_distribution(h):
#     total = h.sum()
#     q = h / total
#     return q

def bootstrap_covariance_test():
    N = 20
    K = 5
    D = 784
    L = 100
    m = 5
    S, h, X = generate_data(sample_count=N, decision_count=K, label_count=L, data_dim=D)
    h = h[:, 0]
    f = 10.0 * np.tan(np.sum(X, axis=1))
    mu_f = np.mean(f)
    mu_h = np.mean(h)
    cov_AB_unbiased_estimate = 0.0
    sample_count = 1000000
    x = range(N)
    all_As_list = []
    all_Bs_list = []
    list_of_lists = [range(N) for k in range(m)]
    for idx in itertools.product(*list_of_lists):
        indices = list(idx)
        A = (f[indices]).mean()
        B = (h[indices]).mean()
        all_As_list.append(A)
        all_Bs_list.append(B)
    A_arr = np.array(all_As_list)
    B_arr = np.array(all_Bs_list)
    cov = np.cov(A_arr, B_arr)

    sample = np.random.choice(x, m, replace=True)




    # A_list = []
    # B_list = []
    # for sample_index in range(sample_count):
    #     sample = np.random.choice(x, m, replace=True)
    #     A = (f[sample]).mean()
    #     B = (h[sample]).mean()
    #     A_list.append(A)
    #     B_list.append(B)
    #     if sample_index % 1000 == 0:
    #         print("mean_A={0}".format(np.array(A_list).mean()))
    #         print("mean_B={0}".format(np.array(B_list).mean()))



    print("X")




def main():
    bootstrap_covariance_test()
    # N = 25
    # K = 5
    # L = 10
    # D = 784
    # n = 1
    # m = 5
    # S, h, X = generate_data(sample_count=N, decision_count=K, label_count=L, data_dim=D)
    # h = h[:,0]
    # log_dict = {}
    # for i in range(n, m+1):
    #     list_of_lists = [range(N) for k in range(i)]
    #     log_list = []
    #     for idx in itertools.product(*list_of_lists):
    #         if len(log_list) % 100000 == 0:
    #             print(len(log_list))
    #         h_selected = h[list(idx)]
    #         avg = np.mean(h_selected)
    #         log_list.append(np.log(avg))
    #     log = np.array(log_list).mean()
    #     log_dict[i] = log
    # print("X")

main()