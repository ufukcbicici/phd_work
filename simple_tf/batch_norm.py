from simple_tf.global_params import GlobalConstants

import tensorflow as tf


def fast_tree_batch_norm(x, masked_x, network, node, decay, iteration, is_training_phase):
    gamma_name = network.get_variable_name(node=node, name="gamma")
    beta_name = network.get_variable_name(node=node, name="beta")
    pop_mean_name = network.get_variable_name(node=node, name="pop_mean")
    pop_var_name = network.get_variable_name(node=node, name="pop_var")
    pop_mean = tf.Variable(name=pop_mean_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                           trainable=False)
    pop_var = tf.Variable(name=pop_var_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
        gamma = tf.Variable(name=gamma_name, initial_value=tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(name=beta_name, initial_value=tf.zeros([x.get_shape()[-1]]))
        node.variablesSet.add(gamma)
        node.variablesSet.add(beta)
    else:
        gamma = None
        beta = None
    mu, sigma = tf.nn.moments(masked_x, [0])
    new_pop_mean = tf.where(iteration > 0, (decay * pop_mean + (1.0 - decay) * mu), mu)
    new_pop_var = tf.where(iteration > 0, (decay * pop_var + (1.0 - decay) * sigma), sigma)
    pop_mean_assign_op = tf.assign(pop_mean, new_pop_mean)
    pop_var_assign_op = tf.assign(pop_var, new_pop_var)

    final_mean = tf.where(is_training_phase > 0, mu, pop_mean)
    final_var = tf.where(is_training_phase > 0, sigma, pop_var)
    normed_masked_x = tf.nn.batch_normalization(x=masked_x, mean=final_mean, variance=final_var, offset=beta,
                                                scale=gamma,
                                                variance_epsilon=1e-5)
    normed_x = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                         variance_epsilon=1e-5)



def batch_norm(x, network, node, decay, iteration, is_decision_phase, is_training_phase):
    gamma_name = network.get_variable_name(node=node, name="gamma")
    beta_name = network.get_variable_name(node=node, name="beta")
    pop_mean_name = network.get_variable_name(node=node, name="pop_mean")
    pop_var_name = network.get_variable_name(node=node, name="pop_var")
    last_mean_name = network.get_variable_name(node=node, name="last_mean")
    last_var_name = network.get_variable_name(node=node, name="last_var")
    pop_mean = tf.Variable(name=pop_mean_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                           trainable=False)
    pop_var = tf.Variable(name=pop_var_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
    last_mean = tf.Variable(name=last_mean_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                            trainable=False)
    last_var = tf.Variable(name=last_var_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                           trainable=False)
    if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
        gamma = tf.Variable(name=gamma_name, initial_value=tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(name=beta_name, initial_value=tf.zeros([x.get_shape()[-1]]))
        node.variablesSet.add(gamma)
        node.variablesSet.add(beta)
    else:
        gamma = None
        beta = None
    mu, sigma = tf.nn.moments(x, [0])
    curr_mean = tf.where(is_decision_phase > 0, mu, last_mean)
    curr_var = tf.where(is_decision_phase > 0, sigma, last_var)
    final_mean = tf.where(is_training_phase > 0, curr_mean, pop_mean)
    final_var = tf.where(is_training_phase > 0, curr_var, pop_var)
    normed_x = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                         variance_epsilon=1e-5)
    with tf.control_dependencies([normed_x]):
        last_mean_assign_op = tf.assign(last_mean, mu)
        last_var_assign_op = tf.assign(last_var, sigma)
        new_pop_mean = tf.where(iteration > 0, (decay * pop_mean + (1.0 - decay) * mu), mu)
        new_pop_var = tf.where(iteration > 0, (decay * pop_var + (1.0 - decay) * sigma), sigma)
        pop_mean_assign_op = tf.assign(pop_mean, new_pop_mean)
        pop_var_assign_op = tf.assign(pop_var, new_pop_var)
    return normed_x, [last_mean_assign_op, last_var_assign_op, pop_mean_assign_op, pop_var_assign_op]

# def batch_norm(x, network, node, decay, iteration, is_decision_phase, is_training_phase):
#     gamma_name = network.get_variable_name(node=node, name="gamma")
#     beta_name = network.get_variable_name(node=node, name="beta")
#     pop_mean_name = network.get_variable_name(node=node, name="pop_mean")
#     pop_var_name = network.get_variable_name(node=node, name="pop_var")
#     if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
#         gamma = tf.Variable(name=gamma_name, initial_value=tf.ones([x.get_shape()[-1]]))
#         beta = tf.Variable(name=beta_name, initial_value=tf.zeros([x.get_shape()[-1]]))
#     else:
#         gamma = None
#         beta = None
#     batch_mean, batch_var = tf.nn.moments(x, [0])
#     pop_mean = tf.Variable(name=pop_mean_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
#     pop_var = tf.Variable(name=pop_var_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]), trainable=False)
#     new_mean = tf.where(iteration > 0, is_decision_phase * (decay * pop_mean + (1.0 - decay) * batch_mean) +
#                         (1.0 - is_decision_phase) * pop_mean, batch_mean)
#     new_var = tf.where(iteration > 0, is_decision_phase * (decay * pop_var + (1.0 - decay) * batch_var) +
#                        (1.0 - is_decision_phase) * pop_var, batch_var)
#     pop_mean_assign_op = tf.assign(pop_mean, new_mean)
#     pop_var_assign_op = tf.assign(pop_var, new_var)
#
#     def get_population_moments_with_update():
#         with tf.control_dependencies([pop_mean_assign_op, pop_var_assign_op]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#
#     final_mean, final_var = tf.cond(is_training_phase > 0,
#                                     get_population_moments_with_update,
#                                     lambda: (tf.identity(pop_mean), tf.identity(pop_var)))
#
#     normed = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
#                                        variance_epsilon=1e-5)
#
#     return normed
#     # return normed, final_mean, final_var, batch_mean, batch_var
