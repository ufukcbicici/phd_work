import tensorflow as tf
import numpy as np

from data_handling.mnist_data_set import MnistDataSet
from simple_tf.global_params import GlobalConstants


class InfoGainLoss:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(prob_distribution):
        log_prob = tf.log(prob_distribution + GlobalConstants.INFO_GAIN_LOG_EPSILON)
        # is_inf = tf.is_inf(log_prob)
        # zero_tensor = tf.zeros_like(log_prob)
        # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
        prob_log_prob = prob_distribution * log_prob
        entropy = -1.0 * tf.reduce_sum(prob_log_prob)
        return entropy, log_prob

    @staticmethod
    def get_loss(p_n_given_x_2d, p_c_given_x_2d, balance_coefficient):
        p_n_given_x_3d = tf.expand_dims(input=p_n_given_x_2d, axis=1)
        p_c_given_x_3d = tf.expand_dims(input=p_c_given_x_2d, axis=2)
        unnormalized_joint_xcn = p_n_given_x_3d * p_c_given_x_3d
        # Calculate p(c,n)
        marginal_p_cn = tf.reduce_mean(unnormalized_joint_xcn, axis=0)
        # Calculate p(n)
        marginal_p_n = tf.reduce_sum(marginal_p_cn, axis=0)
        # Calculate p(c)
        marginal_p_c = tf.reduce_sum(marginal_p_cn, axis=1)
        # Calculate entropies
        entropy_p_cn, log_prob_p_cn = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_cn)
        entropy_p_n, log_prob_p_n = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_n)
        entropy_p_c, log_prob_p_c = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_c)
        # Calculate the information gain
        information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
        information_gain = -1.0 * information_gain
        return information_gain
        # return information_gain, unnormalized_joint_xcn, entropy_p_cn, entropy_p_n, entropy_p_c, \
        #        marginal_p_cn, marginal_p_n, marginal_p_c, log_prob_p_cn, log_prob_p_n, log_prob_p_c

    @staticmethod
    def get_layerwise_loss(layer, balance_coefficient):
        pass



#
# sess = tf.Session()
# prob = np.array([6.0, 0.0, 48.0, 2.0, 63.0, 7.0, 89.0, 10.0, 1.0, 39.0])
# prob = (1.0 / np.sum(prob)) * prob
# p = tf.constant(prob, dtype=GlobalConstants.DATA_TYPE)
# ent = InfoGainLoss.calculate_entropy(prob_distribution=p)
# # a = tf.log(GlobalConstants.INFO_GAIN_LOG_EPSILON)
# x = sess.run([ent])
#
# dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
# N = GlobalConstants.BATCH_SIZE
# K = 2
# L = dataset.get_label_count()
#
# W = tf.Variable(tf.truncated_normal([GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, K],
#                                     stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE), name="W")
# b = tf.Variable(tf.constant(0.0, shape=[K], dtype=GlobalConstants.DATA_TYPE), name="b")
#
# flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
# activations = tf.matmul(flat_data, W) + b
# p_n_given_x_2d = tf.nn.softmax(activations)
# p_c_given_x_2d = GlobalConstants.TRAIN_ONE_HOT_LABELS
# information_gain = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x_2d, p_c_given_x_2d=p_c_given_x_2d)
# # unnormalized_joint_xcn, tf_entropy_p_cn, tf_entropy_p_n, tf_entropy_p_c = \
# optimizer = tf.train.MomentumOptimizer(GlobalConstants.INITIAL_LR, 0.9).minimize(information_gain)
#
#
# init = tf.global_variables_initializer()
# sess.run(init)
# samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
# samples = np.expand_dims(samples, axis=3)
# results = sess.run([unnormalized_joint_xcn, W, b, information_gain, tf_entropy_p_cn, tf_entropy_p_n, tf_entropy_p_c],
#                    feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
#                               GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels})

# tf_unnormalized_joint_xcn = results[0]
# np_W = results[1]
# np_b = results[2]
# tf_info_gain = results[3]
#
# # Numpy implementation
# flat_data = np.reshape(a=samples, newshape=(samples.shape[0], np.prod(samples.shape[1:])))
# branch_activations = np.dot(flat_data, np_W) + np_b
# exp_values = np.exp(branch_activations)
# axis_sum = np.reshape(np.sum(exp_values, axis=1), newshape=(flat_data.shape[0], 1))
# h = exp_values / axis_sum
# p_n_given_x_3d = np.expand_dims(h, axis=1)
# p_c_given_x_3d = np.expand_dims(one_hot_labels, axis=2)
# np_unnormalized_joint_xcn = p_n_given_x_3d * p_c_given_x_3d
# if not np.allclose(tf_unnormalized_joint_xcn, np_unnormalized_joint_xcn):
#     raise Exception("Not same joints!!!")
# # Calculate p(c,n), log p(c,n)
# p_cn = np.mean(np_unnormalized_joint_xcn, axis=0)
# log_p_cn = np.log(p_cn + GlobalConstants.INFO_GAIN_LOG_EPSILON)
# # Calculate p(n), log p(n)
# p_n = np.sum(p_cn, axis=0)
# log_p_n = np.log(p_n + GlobalConstants.INFO_GAIN_LOG_EPSILON)
# # Calculate p(c), log p(c)
# p_c = np.sum(p_cn, axis=1)
# log_p_c = np.log(p_c + GlobalConstants.INFO_GAIN_LOG_EPSILON)
# # Calculate information gain
# entropy_p_cn = -1.0 * np.sum(p_cn * log_p_cn)
# entropy_p_n = -1.0 * np.sum(p_n * log_p_n)
# entropy_p_c = -1.0 * np.sum(p_c * log_p_c)
# np_information_gain = entropy_p_n + entropy_p_c - entropy_p_cn


# Increase info gain
# dataset.reset()
# for i in range(60000):
#     samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=50000)
#     samples = np.expand_dims(samples, axis=3)
#     sess.run([optimizer], feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
#                                       GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels})
#     results = sess.run([information_gain], feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
#                                                       GlobalConstants.TRAIN_ONE_HOT_LABELS: one_hot_labels})
#     ig = results[0]
#     print("IG_{0}={1}".format(i, ig))
#
# print("X")
