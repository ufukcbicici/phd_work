import tensorflow as tf
import numpy as np
from collections import Counter
from simple_tf.cign.fast_tree import FastTreeNetwork

sample_count = 1000
sample_repeat_count = 10000
state_dim = 64
action_space_size = 4
passive_weight = tf.constant(-1e+10)
hiddenLayers = [128, action_space_size]

state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="inputs")
routes_input = tf.placeholder(dtype=tf.int32, shape=[None, action_space_size], name="routes")
policy_selector = tf.placeholder(dtype=tf.int32, shape=[None], name="policy_selector")
rewards_input = tf.placeholder(dtype=tf.float32, shape=[None, action_space_size], name="rewards")

net = state_input
for layer_id, layer_dim in enumerate(hiddenLayers):
    if layer_id < len(hiddenLayers) - 1:
        net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
    else:
        net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
logits = net
passive_weights_matrix = tf.ones_like(routes_input, dtype=tf.float32) * passive_weight
sparse_logits = tf.where(tf.cast(routes_input, tf.bool), logits, passive_weights_matrix)
policies = tf.nn.softmax(sparse_logits)
uniform_distribution = tf.ones_like(logits) * tf.constant(1.0 / action_space_size)
policy_selected = tf.where(tf.cast(policy_selector, tf.bool), policies, uniform_distribution)
# Sample from the policies
policies_tiled = tf.tile(policy_selected, [sample_repeat_count, 1])
sampled_actions = FastTreeNetwork.sample_from_categorical_v2(probs=policies_tiled)

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
    distribution_selector = np.random.choice([0, 1], size=(sample_count, ), p=[0.2, 0.8])

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
                                  policy_selector: distribution_selector})
    final_policy = results[-3]
    final_actions = results[-1]
    sampled_action_distributions = []
    for idx in range(sample_count):
        actions_for_state_idx = [final_actions[idx + j*sample_count] for j in range(sample_repeat_count)]
        counter = Counter(actions_for_state_idx)
        freq_arr = [0] * action_space_size
        for k, v in counter.items():
            freq_arr[k] = v
        freq_arr = np.array(freq_arr)
        freq_arr = freq_arr * (1.0/np.sum(freq_arr))
        print("Route:{0}".format(routes[idx]))
        print("Policy Selection:{0}".format(distribution_selector[idx]))
        print("Policy:{0}".format(final_policy[idx]))
        print("Sampled Policy:{0}".format(freq_arr))
        print("X")
    # Sample actions

    print("X")


if __name__ == "__main__":
    main()
