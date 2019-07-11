from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.mnist_data_set import MnistDataSet
import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants


def batch_norm_2(x, decay, g, b):
    pop_mean = tf.Variable(name="pop_mean", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    last_mean = tf.Variable(name="batch_mean", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                            trainable=False)
    last_var = tf.Variable(name="batch_var", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    mu, sigma = tf.nn.moments(x, [0])
    curr_mean = tf.where(is_decision_phase > 0, mu, last_mean)
    curr_var = tf.where(is_decision_phase > 0, sigma, last_var)
    final_mean = tf.where(is_training_phase > 0, curr_mean, pop_mean)
    final_var = tf.where(is_training_phase > 0, curr_var, pop_var)
    normed_x = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=b, scale=g,
                                         variance_epsilon=1e-5)
    with tf.control_dependencies([normed_x]):
        last_mean_assign_op = tf.assign(last_mean, mu)
        last_var_assign_op = tf.assign(last_var, sigma)
        new_pop_mean = tf.where(iteration > 0, (decay * pop_mean + (1.0 - decay) * mu), mu)
        new_pop_var = tf.where(iteration > 0, (decay * pop_var + (1.0 - decay) * sigma), sigma)
        pop_mean_assign_op = tf.assign(pop_mean, new_pop_mean)
        pop_var_assign_op = tf.assign(pop_var, new_pop_var)

    return normed_x, final_mean, final_var, curr_mean, curr_var, pop_mean, pop_var, mu, sigma, \
           [last_mean_assign_op, last_var_assign_op, pop_mean_assign_op, pop_var_assign_op]


def batch_norm_full(x, decay, g, b):
    # b_mean_2 = tf.Variable(name="batch_mean", initial_value=tf.constant(5.0, shape=[x.get_shape()[-1]]), trainable=False)
    # b_var_2 = tf.Variable(name="batch_var", initial_value=tf.constant(4.0, shape=[x.get_shape()[-1]]), trainable=False)
    # normed_x = tf.nn.batch_normalization(x=x, mean=b_mean_2, variance=b_var_2, offset=b, scale=g, variance_epsilon=1e-5)
    # return normed_x
    mu, sigma = tf.nn.moments(x, [0])
    normed_x = tf.nn.batch_normalization(x=x, mean=mu, variance=sigma, offset=b, scale=g, variance_epsilon=1e-5)
    return normed_x


def batch_norm_eval(x, decay, b_mean, b_var, g, b):
    normed_x = tf.nn.batch_normalization(x=x, mean=b_mean, variance=b_var, offset=b, scale=g, variance_epsilon=1e-5)
    return normed_x


decay = 0.5
dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")

# Network
flat_data = tf.contrib.layers.flatten(GlobalConstants.TRAIN_DATA_TENSOR)
iteration = tf.placeholder(name="iteration", dtype=tf.int64)
is_decision_phase = tf.placeholder(name="is_decision_phase", dtype=tf.int64)
is_training_phase = tf.placeholder(name="is_training_phase", dtype=tf.int64)
if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
    gamma = tf.Variable(name="gamma", initial_value=tf.ones([flat_data.get_shape()[-1]]))
    beta = tf.Variable(name="beta", initial_value=tf.zeros([flat_data.get_shape()[-1]]))
else:
    gamma = None
    beta = None

normed, final_mean_eval, final_var_eval, b_mean_eval, b_var_eval, pop_mean_eval, pop_var_eval, mu_eval, sigma_eval, assign_ops = \
    batch_norm_2(x=flat_data, decay=decay, g=gamma, b=beta)
sum_0 = tf.reduce_sum(normed)
grads_0 = tf.gradients(sum_0, [flat_data, gamma, beta])

normed_full = batch_norm_full(x=flat_data, decay=decay, g=gamma, b=beta)
sum_1 = tf.reduce_sum(normed_full)
grads_1 = tf.gradients(sum_1, [flat_data, gamma, beta])

normed_eval = batch_norm_eval(x=flat_data, decay=decay, b_mean=b_mean_eval, b_var=b_var_eval, g=gamma, b=beta)
sum_2 = tf.reduce_sum(normed_eval)
grads_2 = tf.gradients(sum_2, [flat_data, gamma, beta])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

shadow_mean = 0.0
shadow_var = 0.0
# Training
for epoch in range(5):
    print("***********************Epoch {0}***********************".format(epoch))
    for i in range(10):
        print("Iteration {0}".format(i))
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        # ************************************ Decision ************************************
        eval_list = [final_mean_eval, final_var_eval, b_mean_eval, b_var_eval, pop_mean_eval, pop_var_eval, mu_eval,
                     sigma_eval]
        eval_dict = {v: v for v in eval_list}
        eval_dict["grads_0"] = grads_0
        eval_dict["grads_1"] = grads_1
        eval_dict["sum_0"] = sum_0
        eval_dict["sum_1"] = sum_1
        eval_dict["normed"] = normed
        eval_dict["normed_full"] = normed_full
        eval_dict["assign_ops"] = assign_ops
        results = sess.run(eval_dict,
                           feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                      iteration: i, is_decision_phase: 1, is_training_phase: 1})
        if i == 0:
            shadow_mean = results[b_mean_eval]
            shadow_var = results[b_var_eval]
        else:
            shadow_mean = decay * shadow_mean + (1.0 - decay) * results[b_mean_eval]
            shadow_var = decay * shadow_var + (1.0 - decay) * results[b_var_eval]
        print("")
        print("Decision")
        print("Manual Batch Mean = {0}".format(np.sum(np.mean(samples, axis=0))))
        print("shadow_mean = {0}".format(np.sum(shadow_mean)))
        print("pop_mean_eval = {0}".format(np.sum(results[pop_mean_eval])))
        print("shadow_var = {0}".format(np.sum(shadow_var)))
        print("pop_var_eval = {0}".format(np.sum(results[pop_var_eval])))
        print("b_mean_eval = {0}".format(np.sum(results[b_mean_eval])))
        print("b_var_eval = {0}".format(np.sum(results[b_var_eval])))
        print("mu_eval = {0}".format(np.sum(results[mu_eval])))
        print("sigma_eval = {0}".format(np.sum(results[sigma_eval])))
        print("final_mean_eval = {0}".format(np.sum(results[final_mean_eval])))
        print("final_var_eval = {0}".format(np.sum(results[final_var_eval])))
        dec_mu = np.sum(results[final_mean_eval])
        dec_sigma = np.sum(results[final_var_eval])
        grads_0_np = results["grads_0"]
        grads_1_np = results["grads_1"]
        for g0, g1 in zip(grads_0_np, grads_1_np):
            assert (np.allclose(g0, g1))
        assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_mean), f2=np.sum(results[pop_mean_eval])))
        assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_var), f2=np.sum(results[pop_var_eval])))
        assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_mean_eval]), f2=np.sum(results[mu_eval])))
        assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_var_eval]), f2=np.sum(results[sigma_eval])))
        # ************************************ Decision ************************************

        # ************************************ Classification ************************************
        samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
        samples = np.expand_dims(samples, axis=3)
        eval_list = [final_mean_eval, final_var_eval, b_mean_eval, b_var_eval, pop_mean_eval, pop_var_eval, mu_eval,
                     sigma_eval]
        eval_dict = {v: v for v in eval_list}
        eval_dict["grads_0"] = grads_0
        eval_dict["grads_2"] = grads_2
        results = sess.run(eval_dict,
                           feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                      iteration: i, is_decision_phase: 0, is_training_phase: 1})
        print("")
        print("Classification")
        print("shadow_mean = {0}".format(np.sum(shadow_mean)))
        print("pop_mean_eval = {0}".format(np.sum(results[pop_mean_eval])))
        print("shadow_var = {0}".format(np.sum(shadow_var)))
        print("pop_var_eval = {0}".format(np.sum(results[pop_var_eval])))
        print("b_mean_eval = {0}".format(np.sum(results[b_mean_eval])))
        print("b_var_eval = {0}".format(np.sum(results[b_var_eval])))
        print("mu_eval = {0}".format(np.sum(results[mu_eval])))
        print("sigma_eval = {0}".format(np.sum(results[sigma_eval])))
        print("final_mean_eval = {0}".format(np.sum(results[final_mean_eval])))
        print("final_var_eval = {0}".format(np.sum(results[final_var_eval])))
        grads_0_np = results["grads_0"]
        grads_2_np = results["grads_2"]
        for g0, g2 in zip(grads_0_np, grads_2_np):
            assert (np.allclose(g0, g2))
        assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_mean), f2=np.sum(results[pop_mean_eval])))
        assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_var), f2=np.sum(results[pop_var_eval])))
        assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_mean_eval]), f2=dec_mu))
        assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_var_eval]), f2=dec_sigma))
        # ************************************ Classification ************************************
        #
    # ************************************ Test ************************************
    samples, labels, indices_list, one_hot_labels = dataset.get_next_batch(batch_size=GlobalConstants.BATCH_SIZE)
    samples = np.expand_dims(samples, axis=3)
    eval_list = [final_mean_eval, final_var_eval, b_mean_eval, b_var_eval, pop_mean_eval, pop_var_eval, mu_eval,
                 sigma_eval]
    eval_dict = {v: v for v in eval_list}
    results = sess.run(eval_dict,
                       feed_dict={GlobalConstants.TRAIN_DATA_TENSOR: samples,
                                  iteration: 1000000, is_decision_phase: 0, is_training_phase: 0})
    print("")
    print("Test")
    print("shadow_mean = {0}".format(np.sum(shadow_mean)))
    print("pop_mean_eval = {0}".format(np.sum(results[pop_mean_eval])))
    print("shadow_var = {0}".format(np.sum(shadow_var)))
    print("pop_var_eval = {0}".format(np.sum(results[pop_var_eval])))
    print("b_mean_eval = {0}".format(np.sum(results[b_mean_eval])))
    print("b_var_eval = {0}".format(np.sum(results[b_var_eval])))
    print("mu_eval = {0}".format(np.sum(results[mu_eval])))
    print("sigma_eval = {0}".format(np.sum(results[sigma_eval])))
    print("final_mean_eval = {0}".format(np.sum(results[final_mean_eval])))
    print("final_var_eval = {0}".format(np.sum(results[final_var_eval])))
    assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_mean), f2=np.sum(results[pop_mean_eval])))
    assert (UtilityFuncs.compare_floats(f1=np.sum(shadow_var), f2=np.sum(results[pop_var_eval])))
    assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_mean_eval]), f2=np.sum(results[pop_mean_eval])))
    assert (UtilityFuncs.compare_floats(f1=np.sum(results[final_var_eval]), f2=np.sum(results[pop_var_eval])))
    # ************************************ Test ************************************
print("X")


# def batch_norm(x, decay, g, b):
#     pop_mean = tf.Variable(name="pop_mean", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
#     pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
#     b_mean = tf.Variable(name="batch_mean", initial_value=tf.constant(5.0, shape=[x.get_shape()[-1]]), trainable=False)
#     b_var = tf.Variable(name="batch_var", initial_value=tf.constant(4.0, shape=[x.get_shape()[-1]]), trainable=False)
#     b_mean_2 = tf.Variable(name="batch_mean_2", initial_value=tf.constant(5.0, shape=[x.get_shape()[-1]]), trainable=False)
#     b_var_2 = tf.Variable(name="batch_var_2", initial_value=tf.constant(4.0, shape=[x.get_shape()[-1]]), trainable=False)
#     mu, sigma = tf.nn.moments(x, [0])
#     # decision_b_mean_assign_op = tf.assign(b_mean, is_decision_phase * mu + (1.0 - is_decision_phase) * b_mean)
#     # decision_b_var_assign_op = tf.assign(b_var, is_decision_phase * sigma + (1.0 - is_decision_phase) * b_var)
#     decision_b_mean_assign_op = tf.assign(b_mean, mu)
#     decision_b_var_assign_op = tf.assign(b_var, sigma)
#     with tf.control_dependencies([decision_b_mean_assign_op, decision_b_var_assign_op]):
#         new_mean = tf.where(iteration > 0, is_decision_phase * (decay * pop_mean + (1.0 - decay) * b_mean) + (
#         1.0 - is_decision_phase) * pop_mean, b_mean)
#         new_var = tf.where(iteration > 0, is_decision_phase * (decay * pop_var + (1.0 - decay) * b_var) + (
#         1.0 - is_decision_phase) * pop_var, b_var)
#         pop_mean_assign_op = tf.assign(pop_mean, new_mean)
#         pop_var_assign_op = tf.assign(pop_var, new_var)
#
#         def get_population_moments_with_update():
#             with tf.control_dependencies([pop_mean_assign_op, pop_var_assign_op]):
#                 return tf.identity(b_mean), tf.identity(b_var)
#
#         final_mean, final_var = tf.cond(is_training_phase > 0, get_population_moments_with_update,
#                                         lambda: (tf.identity(pop_mean), tf.identity(pop_var)))
#         normed_x = tf.nn.batch_normalization(x=x, mean=tf.identity(mu), variance=tf.identity(sigma), offset=b, scale=g,
#                                              variance_epsilon=1e-5)
#         return normed_x, final_mean, final_var, b_mean, b_var, pop_mean, pop_var, mu, sigma
#

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
