import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.special import logsumexp
import pdb
import warnings
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


def gumbel_softmax_density(z, probs, temperature):
    n = z.shape[0]
    a = np.math.factorial(n - 1)
    b = np.power(temperature, n - 1)
    z_pow_minus_lambda = np.power(z, -temperature)
    z_pow_minus_lambda_minus_one = np.power(z, -temperature - 1.0)
    numerator_vec = np.multiply(probs, z_pow_minus_lambda_minus_one)
    denominator_vec = np.multiply(probs, z_pow_minus_lambda)
    denominator_sum = np.sum(denominator_vec)
    numerator_vec = numerator_vec / denominator_sum
    c = np.prod(numerator_vec)
    density = a * b * c
    return density


def gumbel_softmax_density_1d(x, probs, temperature):
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    ba = b - a
    z = (x * ba) + a
    n = probs.shape[0]
    a = np.math.factorial(n - 1)
    b = np.power(temperature, n - 1)
    z_pow_minus_lambda = np.power(z, -temperature)
    z_pow_minus_lambda_minus_one = np.power(z, -temperature - 1.0)
    numerator_vec = np.multiply(probs, z_pow_minus_lambda_minus_one)
    denominator_vec = np.multiply(probs, z_pow_minus_lambda)
    denominator_sum = np.sum(denominator_vec)
    numerator_vec = numerator_vec / denominator_sum
    c = np.prod(numerator_vec)
    density = a * b * c
    return density


def gumbel_softmax_density_2d(y, x, probs, temperature):
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    assert y + x <= 1.0
    ca = c - a
    ba = b - a
    z = a + x*ca + y*ba
    assert np.isclose(np.sum(z), np.array([1.0]))
    n = probs.shape[0]
    a = np.math.factorial(n - 1)
    b = np.power(temperature, n - 1)
    z_pow_minus_lambda = np.power(z, -temperature)
    z_pow_minus_lambda_minus_one = np.power(z, -temperature - 1.0)
    numerator_vec = np.multiply(probs, z_pow_minus_lambda_minus_one)
    denominator_vec = np.multiply(probs, z_pow_minus_lambda)
    denominator_sum = np.sum(denominator_vec)
    numerator_vec = numerator_vec / denominator_sum
    c = np.prod(numerator_vec)
    density = a * b * c
    return density


def gumbel_softmax_density_v2(x, alpha, temperature):
    n = x.shape[0]
    a = np.math.factorial(n - 1)
    b = np.power(temperature, n - 1)
    denom = 0.0
    for _i in range(n):
        denom += alpha[_i]*(x[_i]**(-temperature))
    c = 1.0
    for _i in range(n):
        c *= (alpha[_i]*(x[_i]**(-temperature-1.0)))/denom
    return a*b*c

probs2d = np.array([0.6, 0.4])
probs3d = np.array([0.6, 0.3, 0.1])
temperature = 0.1

uniform_samples = np.random.uniform(low=0.0, high=1.0, size=(10000, 3))
gumbel_samples = -1.0 * np.log(-1.0 * np.log(uniform_samples))
# Concrete
log_probs = np.log(probs3d)
pre_transform = log_probs + gumbel_samples
temp_divided = pre_transform / temperature
logits = np.exp(temp_divided)
nominator = np.expand_dims(np.sum(logits, axis=1), axis=1)
z_samples = logits / nominator
log_sum_exp = np.expand_dims(logsumexp(temp_divided, axis=1), axis=1)
y_samples = temp_divided - log_sum_exp
z_samples_stable = np.exp(y_samples)
print("X")
# log_probs = tf.expand_dims(log_probs, dim=1)
# pre_transform = log_probs + gumbel_sample
# temp_divided = pre_transform / temperature_tensor
# logits = tf.math.exp(temp_divided)
# nominator = tf.expand_dims(tf.reduce_sum(logits, axis=2), dim=2)
# z_samples = logits / nominator
















# a = np.array([1.0, 0.0])
# b = np.array([0.0, 1.0])
# ba = b - a
# epsilon = 0.0000001
# volume = np.linalg.norm(ba)
#
# I = quad(gumbel_softmax_density_1d, 0, 1, args=(probs2d, 0.5))
# I0 = quad(gumbel_softmax_density_1d, 0, 0.5, args=(probs2d, 0.5))
# I1 = quad(gumbel_softmax_density_1d, 0.5, 1.0, args=(probs2d, 0.5))
# I = quad(gumbel_softmax_density_1d, 0, 1, args=(probs2d, 0.25))
# I0 = quad(gumbel_softmax_density_1d, 0, 0.5, args=(probs2d, 0.25))
# I1 = quad(gumbel_softmax_density_1d, 0.5, 1.0, args=(probs2d, 0.25))
#
#
I = dblquad(gumbel_softmax_density_2d, 0, 1, gfun=(lambda x: 0), hfun=(lambda x: 1.0-x), args=(probs3d, 1.0))

interval_count = 50000000
upper_limit = 1.0
lower_limit = 0.0
interval_length = (upper_limit - lower_limit) / interval_count
curr_x = lower_limit
densities = []
for i in range(interval_count - 1):
    curr_x += interval_length
    density = gumbel_softmax_density_1d(x=curr_x, probs=probs2d, temperature=0.25)
    densities.append(density)
integral = (upper_limit - lower_limit) * (1.0 / len(densities)) * sum(densities)
print("integral={0}".format(integral))
# counter = 0.0
# list_of_simplex_points = []
# dict_of_samples = {}
# while True:
#     counter += 1.0
#     p = counter*epsilon*ba + a
#     if p[0] <= 0.0:
#         break
#     list_of_simplex_points.append(p)
#
# for p in list_of_simplex_points:
#     density = gumbel_softmax_density(z=p, probs=probs, temperature=0.5)
#     index = np.argmax(p)
#     if index not in dict_of_samples:
#         dict_of_samples[index] = []
#     dict_of_samples[index].append(density)
#
# integrations = {}
# for k, v in dict_of_samples.items():
#     integration = (volume / 2.0) * sum(dict_of_samples[k]) / len(dict_of_samples[k])
#     integrations[k] = integration
#     print("Integral {0}:{1}".format(k, integration))
#
# print(integrations[0] / integrations[1])


















    # n = z.shape[0]
    # a = np.math.factorial(n)
    # b = np.power(temperature, n - 1)
    # z_pow_minus_lambda = np.power(z, -temperature)
    # z_pow_minus_lambda_minus_one = np.power(z, -temperature - 1.0)
    # numerator_vec = np.multiply(probs, z_pow_minus_lambda_minus_one)
    # denominator_vec = np.multiply(probs, z_pow_minus_lambda)
    # denominator_sum = np.sum(denominator_vec)
    # numerator_vec = numerator_vec / denominator_sum
    # c = np.prod(numerator_vec)
    # density = a * b * c
    # return density





# Draw samples on the 2-simplex (triangle) consisting of (1,0,0),(0,1,0) and (0,0,1)
# a = np.array([1.0, 0.0, 0.0])
# b = np.array([0.0, 1.0, 0.0])
# c = np.array([0.0, 0.0, 1.0])
# ba = b-a
# ca = c-a
#
# # density = gumbel_softmax_density(z=np.array([0.9, 0.04, 0.06]), probs=np.array([0.5, 0.3, 0.2]), temperature=0.1)
# # density2 = gumbel_softmax_density_v2(x=np.array([0.9, 0.04, 0.06]), alpha=np.array([0.5, 0.3, 0.2]), temperature=0.1)
# epsilon = 0.01
# u = 0.0
# list_of_points = []
# while True:
#     if u > 1.0:
#         break
#     v = 0.0
#     while u + v <= 1.0:
#         p = a + u*ba + v*ca
#         list_of_points.append(p)
#         v += epsilon
#     u += epsilon
# p_list = np.array(list_of_points)
# p_closeness_dict = {}
# for index, p in enumerate(p_list):
#     if index % 100000 == 0:
#         print(index)
#     i = np.argmax(p)
#     if i not in p_closeness_dict:
#         p_closeness_dict[i] = []
#     p_closeness_dict[i].append(p)
# print(len(p_closeness_dict[0]))
# print(len(p_closeness_dict[1]))
# print(len(p_closeness_dict[2]))
#
# probs = np.array([0.3, 0.5, 0.2])
# total_samples_dict = {}
# sample_count_dict = {}
# for k, v in p_closeness_dict.items():
#     total_samples_dict[k] = 0.0
#     sample_count_dict[k] = 0.0
#     for p in v:
#         density = gumbel_softmax_density(z=p, probs=probs, temperature=0.25)
#         if np.isnan(density):
#             continue
#         print("p={0} density={1}".format(p, density))
#         total_samples_dict[k] += density
#         sample_count_dict[k] += 1.0
# for k in total_samples_dict.keys():
#     print("{0} - {1}".format(k, total_samples_dict[k]/sample_count_dict[k]))
# print("X")

    # if np.abs(np.sum(anchor_u) - 1.0) > epsilon / 2.0:
    #     break
    # anchor_u += epsilon*u
    # p = anchor_u.copy()
    # while True:
    #     if np.abs(np.sum(p) - 1.0) > epsilon/2.0:
    #         break
    #     list_of_points.append(p)
    #     p += epsilon * v

