from data_handling.mnist_data_set import MnistDataSet
import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants


def batch_norm(x, decay):
    if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
        gamma = tf.Variable(name="gamma", initial_value=tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(name="beta", initial_value=tf.zeros([x.get_shape()[-1]]))
    else:
        gamma = None
        beta = None
    pop_mean = tf.Variable(name="pop_mean", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    decision_batch_mean = tf.Variable(name="decision_batch_mean",
                                      initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    decision_batch_var = tf.Variable(name="decision_batch_var",
                                     initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, [0])
    decision_batch_mean_assign_op = tf.assign(decision_batch_mean,
                                              is_decision_phase * batch_mean +
                                              (1.0 - is_decision_phase) * decision_batch_mean)
    decision_batch_var_assign_op = tf.assign(decision_batch_var, is_decision_phase * batch_var +
                                             (1.0 - is_decision_phase) * decision_batch_var)
    with tf.control_dependencies([decision_batch_mean_assign_op, decision_batch_var_assign_op]):
        new_mean = tf.where(iteration > 0, is_decision_phase * (decay * pop_mean + (1.0 - decay)
                                                                * decision_batch_mean) +
                            (1.0 - is_decision_phase) * pop_mean, decision_batch_mean)
        new_var = tf.where(iteration > 0, is_decision_phase * (decay * pop_var + (1.0 - decay) * decision_batch_var) +
                           (1.0 - is_decision_phase) * pop_var, decision_batch_var)
        pop_mean_assign_op = tf.assign(pop_mean, new_mean)
        pop_var_assign_op = tf.assign(pop_var, new_var)

        def get_population_moments_with_update():
            with tf.control_dependencies([pop_mean_assign_op, pop_var_assign_op]):
                return tf.identity(decision_batch_mean), tf.identity(decision_batch_var)
        final_mean, final_var = tf.cond(is_training_phase > 0,
                                        get_population_moments_with_update,
                                        lambda: (tf.identity(pop_mean), tf.identity(pop_var)))
        normed = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                           variance_epsilon=1e-5)
        return normed, final_mean, final_var, batch_mean, batch_var

decay = 0.5
dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")

# Network
iteration = tf.placeholder(name="iteration", dtype=tf.int64)
is_decision_phase = tf.placeholder(name="is_decision_phase", dtype=tf.float32)
is_training_phase = tf.placeholder(name="is_training_phase", dtype=tf.int64)
flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
normed, pop_mean_eval, pop_var_eval, batch_mean, batch_var = batch_norm(x=flat_data, decay=decay)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

shadow_mean = 0.0
shadow_var = 0.0
# Training
for epoch in range(5):
    for i in range(10):
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        print("Epoch {0}".format(epoch))
        # ************************************ Decision ************************************
        results = sess.run([pop_mean_eval, pop_var_eval, batch_mean, batch_var, normed],
                           feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                      iteration: i, is_decision_phase: 1.0, is_training_phase: 1})
        if i == 0:
            shadow_mean = results[2]
            shadow_var = results[3]
        else:
            shadow_mean = decay * shadow_mean + (1.0 - decay) * results[2]
            shadow_var = decay * shadow_var + (1.0 - decay) * results[3]
        print("")
        print("Decision")
        print("Manual Ema (Mean) {0} = {1}".format(i, np.sum(shadow_mean)))
        print("Auto Ema (Mean) {0} = {1}".format(i, np.sum(results[0])))
        print("Manual Ema (Var) {0} = {1}".format(i, np.sum(shadow_var)))
        print("Auto Ema (Var) {0} = {1}".format(i, np.sum(results[1])))
        # ************************************ Decision ************************************

        # ************************************ Classification ************************************
        results = sess.run([pop_mean_eval, pop_var_eval, batch_mean, batch_var, normed],
                           feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                      iteration: i, is_decision_phase: 0.0, is_training_phase: 1})
        print("")
        print("Classification")
        print("Auto Ema (Mean) {0} = {1}".format(i, np.sum(results[0])))
        print("Auto Ema (Var) {0} = {1}".format(i, np.sum(results[1])))
        # ************************************ Classification ************************************
        #
    # ************************************ Test ************************************
    samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
    samples = np.expand_dims(samples, axis=3)
    results = sess.run([pop_mean_eval, pop_var_eval, batch_mean, batch_var, normed],
                       feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                  iteration: 1000000, is_decision_phase: 0.0, is_training_phase: 0})
    print("")
    print("Test")
    print("Auto Ema (Mean) {0} = {1}".format(i, np.sum(results[0])))
    print("Auto Ema (Var) {0} = {1}".format(i, np.sum(results[1])))
    print("batch_mean {0} = {1}".format(i, np.sum(results[2])))
    print("batch_var {0} = {1}".format(i, np.sum(results[3])))
    # ************************************ Test ************************************
print("X")


# decay = 0.5
# dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
# flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
# batch_mean, batch_var = tf.nn.moments(flat_data, [0])
# ema = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)
# single_entry = batch_mean[70]
# ema_apply_op = ema.apply([single_entry])
# ema_mean = ema.average(single_entry)
# with tf.control_dependencies([ema_apply_op]):
#     ema_mean_eval = tf.identity(ema_mean)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# shadow_mean = 0.0
# for i in range(10):
#     samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
#     samples = np.expand_dims(samples, axis=3)
#     results = sess.run([ema_mean_eval, single_entry], feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples})
#     if i == 0:
#         shadow_mean = results[1]
#     else:
#         shadow_mean -= (1.0 - decay) * (shadow_mean - results[1])
#     print("")
#     print("Manual Ema {0} = {1}".format(i, shadow_mean))
#     print("Auto Ema {0} = {1}".format(i, results[0]))
# print("X")


# dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
# is_train_phase = tf.placeholder(name="is_train", dtype=tf.bool)
# is_decision_phase = tf.placeholder(name="is_decision_phase", dtype=tf.bool)
#
# x = tf.Variable(name="x", initial_value=1.0, trainable=False)
# y = tf.Variable(name="y", initial_value=2.0, trainable=False)
# flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
#
#
# batch_mean, batch_var = tf.nn.moments(flat_data, [0])
# ema = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)
# ema_apply_op = ema.apply([batch_mean, batch_var])
# pop_mean = ema.average(batch_mean)
# pop_var = ema.average(batch_var)
#
#
# def use_batch_mean_with_update():
#     with tf.control_dependencies([ema_apply_op]):
#         pop_mean_eval = tf.identity(pop_mean)
#         pop_var_eval = tf.identity(pop_var)
#         z = tf.constant(5)
#         return pop_mean_eval, pop_var_eval, z
#
#
# def use_batch_mean_without_update():
#     pop_mean_eval = tf.identity(pop_mean)
#     pop_var_eval = tf.identity(pop_var)
#     z = tf.constant(3)
#     return pop_mean_eval, pop_var_eval, z
#
# pop_mean_result, pop_var_result, res_z = tf.cond(is_decision_phase, use_batch_mean_with_update, use_batch_mean_without_update)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# shadow_mean = 0.0
# shadow_var = 0.0
# for i in range(10):
#     samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
#     samples = np.expand_dims(samples, axis=3)
#     results = sess.run([pop_mean_result, batch_mean, pop_var_result, batch_var, res_z],
#                        feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples, is_decision_phase: True})
#     print("***************************")
#     if i == 0:
#         shadow_mean = results[1][70]
#         shadow_var = results[3][70]
#     else:
#         shadow_mean -= (1.0 - decay) * (shadow_mean - results[1][70])
#         shadow_var -= (1.0 - decay) * (shadow_var - results[3][70])
#         # shadow_mean = decay * shadow_mean + (1.0 - decay) * results[1][70]
#         # shadow_var = decay * shadow_var + (1.0 - decay) * results[3][70]
#     print("pop_mean_result={0}".format(results[0][70]))
#     print("batch_mean={0}".format(results[1][70]))
#     print("pop_var_result={0}".format(results[2][70]))
#     print("batch_var={0}".format(results[3][70]))
#     print(results[4])
#     print("shadow_mean={0}".format(shadow_mean))
#     print("shadow_var={0}".format(shadow_var))
#     print("***************************")
#     results = sess.run([pop_mean_result, batch_mean, pop_var_result, batch_var, res_z],
#                        feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples, is_decision_phase: False})
#     print("***************************")
#     print("pop_mean_result={0}".format(results[0][70]))
#     print("batch_mean={0}".format(results[1][70]))
#     print("pop_var_result={0}".format(results[2][70]))
#     print("batch_var={0}".format(results[3][70]))
#     print(results[4])
#     print("shadow_mean={0}".format(shadow_mean))
#     print("shadow_var={0}".format(shadow_var))
#     print("***************************")




# a = tf.Variable(name="a", initial_value=5.0, trainable=False)
# b = tf.Variable(name="b", initial_value=4.0, trainable=False)
# c = tf.Variable(name="c", initial_value=0.0, trainable=False)
# flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
# pop_mean = tf.Variable(name="mean", initial_value=tf.constant(0.0, shape=[flat_data.get_shape()[-1]]))
# pop_var = tf.Variable(name="var", initial_value=tf.constant(0.0, shape=[flat_data.get_shape()[-1]]))
# batch_mean, batch_var = tf.nn.moments(flat_data, [0])
#
# b_op = tf.assign(b, a)
# mean_op = tf.assign(pop_mean, batch_mean)
# with tf.control_dependencies([b_op, mean_op]):
#     c = a + b
#     pop_mean_current = tf.identity(pop_mean)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# results = sess.run([c, pop_mean_current], feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples})
# print("X")


# population_mean = tf.Variable(name="mean", initial_value=tf.constant(100.0, shape=[flat_data.get_shape()[-1]],
#                                                                      dtype=tf.float32), trainable=False,
#                                                                         dtype=tf.float32)
# a = tf.Variable(name="a", initial_value=5.0, trainable=False)
# b = tf.Variable(name="b", initial_value=4.0, trainable=False)
# c = tf.Variable(name="c", initial_value=0.0, trainable=False)


# def batch_norm_wrapper(inputs, is_train_flag, is_decision_flag, decay=0.9):
#     if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
#         gamma = tf.Variable(name="gamma", initial_value=tf.ones([inputs.get_shape()[-1]]))
#         beta = tf.Variable(name="beta", initial_value=tf.zeros([inputs.get_shape()[-1]]))
#     else:
#         gamma = None
#         beta = None
#     pop_mean = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
#     pop_var = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
#     batch_mean, batch_var = tf.nn.moments(inputs, [0])
#     train_mean_assign_op = tf.assign(pop_mean, batch_mean)
#     train_var_assign_op = tf.assign(pop_var, batch_var)
#     with tf.control_dependencies([train_mean_assign_op, train_var_assign_op]):
#         foo = tf.Variable(name="foo", initial_value=3.0)
#         zoo = foo + 5.0
#         return pop_mean, pop_var, batch_mean, batch_var, zoo
#     # def mean_var_with_update():
#     #     # train_mean_assign_op = tf.cond(is_decision_flag,
#     #     #                                lambda: tf.assign(pop_mean, pop_mean * decay + batch_mean * (1.0 - decay)),
#     #     #                                lambda: tf.assign(pop_mean, pop_mean))
#     #     # train_var_assign_op = tf.cond(is_decision_flag,
#     #     #                               lambda: tf.assign(pop_var, pop_var * decay + batch_var * (1.0 - decay)),
#     #     #                               lambda: tf.assign(pop_var, pop_var))
#     #     train_mean_assign_op = tf.assign(pop_mean, batch_mean)
#     #     train_var_assign_op = tf.assign(pop_var, batch_var)
#     #     with tf.control_dependencies([train_mean_assign_op, train_var_assign_op]):
#     #         return batch_mean, batch_var
#     #
#     # mean, var = tf.cond(is_train_flag, mean_var_with_update, lambda: (pop_mean, pop_var))
#     # normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
#     # return pop_mean, pop_var, mean, var
#
#
# is_decision = tf.placeholder(name="is_decision", dtype=tf.bool)
# is_train = tf.placeholder(name="is_train", dtype=tf.bool)
#
# dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
#
# hyperplane_weights = tf.Variable(
#     tf.truncated_normal([GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, 2],
#                         stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
#     name="hyperplane_weights")
# hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[2], dtype=GlobalConstants.DATA_TYPE),
#                                 name="hyperplane_biases")
# flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
# pop_mean, pop_mean_var, curr_mean, curr_var, zoo = batch_norm_wrapper(inputs=flat_data, is_train_flag=is_train,
#                                                                               is_decision_flag=is_decision)
# normed_data = tf.identity(flat_data)
# activations = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
# normed_activations = tf.matmul(normed_data, hyperplane_weights) + hyperplane_biases
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
#
# samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
# samples = np.expand_dims(samples, axis=3)
#
# sess.run(init)
#
# results = sess.run([normed_activations, activations, flat_data, normed_data, pop_mean, pop_mean_var,
#                     curr_mean, curr_var, zoo],
#                    feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples, is_train: True, is_decision: True})
# res_normed_activations = results[0]
# res_activations = results[1]
# res_flat_data = results[2]
# res_normed_data = results[3]
# res_pop_mean = results[4]
# res_pop_mean_var = results[5]
# res_curr_mean = results[6]
# res_curr_var = results[7]
# res_zoo = results[8]
# print("X")
