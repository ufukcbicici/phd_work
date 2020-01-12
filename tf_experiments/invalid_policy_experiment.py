import tensorflow as tf
import numpy as np
from collections import Counter

from simple_tf.cign.fast_tree import FastTreeNetwork

sample_count = 25
sample_repeat_count = 100
state_dim = 10
action_space_size = 4
passive_weight = tf.constant(-1e+10)
epsilon_prob = tf.constant(1e-3)
hiddenLayers = [128, action_space_size]

state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="inputs")
routes_input = tf.placeholder(dtype=tf.int32, shape=[None, action_space_size], name="routes")
policy_selector = tf.placeholder(dtype=tf.int32, shape=[None], name="policy_selector")
rewards_input = tf.placeholder(dtype=tf.float32, shape=[None, action_space_size], name="rewards")
sampled_actions_input = tf.placeholder(dtype=tf.int32, shape=[None], name="sampled_actions_input")
sample_repeat_count_input = tf.placeholder(dtype=tf.int32, name="sample_repeat_count_input")

net = state_input
for layer_id, layer_dim in enumerate(hiddenLayers):
    if layer_id < len(hiddenLayers) - 1:
        net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
    else:
        net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
logits = net
passive_weights_matrix = tf.ones_like(routes_input, dtype=tf.float32) * passive_weight
sparse_logits = tf.where(tf.cast(routes_input, tf.bool), logits, passive_weights_matrix)
policies_non_modified = tf.nn.softmax(sparse_logits)
policies = policies_non_modified + tf.where(tf.cast(routes_input, tf.bool),
                                            tf.zeros_like(policies_non_modified, dtype=tf.float32),
                                            tf.ones_like(policies_non_modified, dtype=tf.float32) * epsilon_prob)
uniform_distribution = tf.ones_like(logits) * tf.constant(1.0 / action_space_size)
policy_selected = tf.where(tf.cast(policy_selector, tf.bool), policies, uniform_distribution)
# Sample from the policies
policies_tiled = tf.tile(policy_selected, [sample_repeat_count_input, 1])
rewards_tiled = tf.tile(rewards_input, [sample_repeat_count_input, 1])
log_policies_tiled = tf.log(policies_tiled)
sampled_actions = FastTreeNetwork.sample_from_categorical_v2(probs=policies_tiled)
# Build Proxy Loss
prob_shape = tf.shape(policies_tiled)
num_states = tf.gather_nd(prob_shape, [0])
num_categories = tf.gather_nd(prob_shape, [1])
sampled_actions_2d = tf.stack([tf.range(0, num_states, 1), sampled_actions], axis=1)
log_sampled_policies = tf.gather_nd(log_policies_tiled, sampled_actions_2d)
sampled_rewards = tf.gather_nd(rewards_tiled, sampled_actions_2d)
trajectories = log_sampled_policies * sampled_rewards
proxy_policy_value = tf.reduce_mean(trajectories)
grads = tf.gradients(proxy_policy_value, [log_policies_tiled,
                                          policies_tiled,
                                          policy_selected,
                                          policies,
                                          policies_non_modified,
                                          sparse_logits,
                                          logits])


# uniform_distribution = tf.ones_like(logits) * tf.constant(1.0 / action_space_size)
# policy = tf.nn.softmax(logits)
# policy_selected = tf.where(tf.cast(policy_selector, tf.bool), policy, uniform_distribution)

# passive_weights_matrix = tf.ones_like(routes_input, dtype=tf.float32) * passive_weight
# sparse_logits = tf.where(tf.cast(routes_input, tf.bool),  logits, passive_weights_matrix)
# policies = tf.nn.softmax(sparse_logits)
# log_policies = tf.log(policies)
# value_matrix = log_policies * rewards_input
# policy_value = tf.reduce_mean(value_matrix)
# grads = tf.gradients(policy_value, [log_policies, policies, sparse_logits, logits])


def main():
    # Create artificial data
    states = np.random.uniform(size=(sample_count, state_dim))
    routes = np.random.choice([0, 1], size=(sample_count, action_space_size), p=[0.1, 0.9])
    rewards = np.random.uniform(low=-5.0, high=5.0, size=(sample_count, action_space_size))
    distribution_selector = np.random.choice([0, 1], size=(sample_count,), p=[0.2, 0.8])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Simulate Policy Sampling
    results = sess.run([logits,
                        sparse_logits,
                        policies,
                        policy_selected,
                        policies_tiled,
                        sampled_actions],
                       feed_dict={state_input: states,
                                  routes_input: routes,
                                  rewards_input: rewards,
                                  policy_selector: distribution_selector,
                                  sample_repeat_count_input: sample_repeat_count})
    final_policy = results[-3]
    final_actions = results[-1]

    # sampled_action_distributions = []
    # for idx in range(sample_count):
    #     actions_for_state_idx = [final_actions[idx + j*sample_count] for j in range(sample_repeat_count)]
    #     counter = Counter(actions_for_state_idx)
    #     freq_arr = [0] * action_space_size
    #     for k, v in counter.items():
    #         freq_arr[k] = v
    #     freq_arr = np.array(freq_arr)
    #     freq_arr = freq_arr * (1.0/np.sum(freq_arr))
    #     print("Route:{0}".format(routes[idx]))
    #     print("Policy Selection:{0}".format(distribution_selector[idx]))
    #     print("Policy:{0}".format(final_policy[idx]))
    #     print("Sampled Policy:{0}".format(freq_arr))
    #     print("X")

    # Simulate Proxy Loss
    results = sess.run([logits,
                        sparse_logits,
                        policies_non_modified,
                        policies,
                        policy_selected,
                        policies_tiled,
                        sampled_actions_2d,
                        rewards_tiled,
                        log_policies_tiled,
                        log_sampled_policies,
                        proxy_policy_value,
                        grads],
                       feed_dict={state_input: states,
                                  routes_input: routes,
                                  rewards_input: rewards,
                                  policy_selector: distribution_selector,
                                  sample_repeat_count_input: sample_repeat_count})
    log_policy = results[-2]
    log_sampled_policy = results[-3]
    _sampled_actions = results[6]
    grads_arr = results[-1]
    is_inf_array = np.isinf(log_sampled_policy)
    assert not np.any(is_inf_array)
    # dy/d(log_policies_tiled) -> Only selected actions are nonzero.
    selection_arr = np.zeros(shape=(_sampled_actions.shape[0], action_space_size))
    selection_arr[_sampled_actions[:, 0], _sampled_actions[:, 1]] = 1
    assert np.array_equal(selection_arr, (grads_arr[0] != 0.0).astype(np.int32))
    # dy/d(policies_tiled) -> Only selected actions are nonzero.
    assert np.array_equal(selection_arr, (grads_arr[1] != 0.0).astype(np.int32))
    # dy/d(policy_selected) -> Sum of all tiled policies_tiled elements.
    tiled_grads = [grads_arr[1][_i*sample_count:(_i+1)*sample_count] for _i in range(sample_repeat_count)]
    sum_tiled_grads = np.sum(np.stack(tiled_grads, axis=len(tiled_grads[0].shape)), axis=-1)
    assert np.allclose(grads_arr[2], sum_tiled_grads)
    # dy/d(policies) -> Whenever distribution_selector[row]=1 then dy/d(policies)[row] = dy/d(policy_selected)[row]
    # Else: distribution_selector[row]=0
    dy_d_policies_selected = np.copy(grads_arr[2])
    assert np.array_equal(grads_arr[3], dy_d_policies_selected * np.expand_dims(distribution_selector, 1))
    print("X")


# grads = tf.gradients(proxy_policy_value, [log_policies_tiled,
#                                           policies_tiled,
#                                           policy_selected,
#                                           policies,
#                                           policies_non_modified,
#                                           sparse_logits,
#                                           logits])


if __name__ == "__main__":
    main()
