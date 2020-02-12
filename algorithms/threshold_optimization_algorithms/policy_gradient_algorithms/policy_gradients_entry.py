import tensorflow as tf
import numpy as np
import time

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.full_gpu_tree_policy_network import \
    FullGpuTreePolicyGradientsNetwork
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.full_gpu_tree_policy_network_valid_actions import \
    FullGpuTreePolicyTreeNetworkValidActions
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.tree_depth_policy_network import \
    TreeDepthPolicyNetwork
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


# class TreeMlpPolicyNetwork(FullGpuTreePolicyGradientsNetwork):
#     def __init__(self, validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
#                  output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
#                  trajectory_per_state_sample_count):
#         super().__init__(validation_data, test_data, l2_lambda, network, network_name, run_id, iteration, degree_list,
#                          output_names, used_feature_names, hidden_layers, use_baselines, state_sample_count,
#                          trajectory_per_state_sample_count)
#
#     def build_policy_networks(self, time_step):
#         # Create the policy network
#         hidden_layers = list(self.hiddenLayers[time_step])
#         hidden_layers.append(self.actionSpaces[time_step].shape[0])
#         net = self.stateInputTransformed[time_step]
#         for layer_id, layer_dim in enumerate(hidden_layers):
#             if layer_id < len(hidden_layers) - 1:
#                 net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
#             else:
#                 net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
#         _logits = net
#         self.logits.append(_logits)
#         self.policies.append(tf.nn.softmax(_logits / self.softmaxDecay))
#         self.logPolicies.append(tf.log(self.policies[-1]))
#         self.resultsDict["logits_{0}".format(time_step)] = self.logits[time_step]
#         self.resultsDict["policies_{0}".format(time_step)] = self.policies[time_step]
#         self.resultsDict["logPolicies_{0}".format(time_step)] = self.logPolicies[time_step]


def compare_gpu_implementation():
    run_id = 453
    network_name = "FashionNet_Lite"
    iteration = 43680

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                             output_names=output_names)
    validation_data, test_data = routing_data.apply_validation_test_split(test_ratio=0.1)

    wd_list = [0.0] * 10
    # [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005]
    state_sample_count_list = [1000]
    samples_per_state_list = [1]
    ignore_invalid_actions = False
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[wd_list,
                                                                          state_sample_count_list,
                                                                          samples_per_state_list])

    for tpl in cartesian_product:
        sess = tf.Session()
        l2_wd = tpl[0]
        state_sample_count = tpl[1]
        samples_per_state = tpl[2]
        with tf.variable_scope("test"):
            gpu_policy_grads = \
                FullGpuTreePolicyGradientsNetwork(l2_lambda=l2_wd,
                                                  network=network,
                                                  network_name=network_name,
                                                  run_id=run_id,
                                                  iteration=iteration,
                                                  degree_list=[2, 2],
                                                  output_names=output_names,
                                                  used_feature_names=used_output_names,
                                                  use_baselines=True,
                                                  state_sample_count=state_sample_count,
                                                  trajectory_per_state_sample_count=samples_per_state,
                                                  hidden_layers=[[128], [256]],
                                                  validation_data=validation_data,
                                                  test_data=test_data)
        with tf.variable_scope("test", reuse=True):
            cpu_policy_grads = TreeDepthPolicyNetwork(l2_lambda=l2_wd,
                                                      network=network,
                                                      network_name=network_name,
                                                      run_id=run_id,
                                                      iteration=iteration,
                                                      degree_list=[2, 2],
                                                      output_names=output_names,
                                                      used_feature_names=used_output_names,
                                                      use_baselines=True,
                                                      state_sample_count=state_sample_count,
                                                      trajectory_per_state_sample_count=samples_per_state,
                                                      hidden_layers=[[128], [256]],
                                                      validation_data=validation_data,
                                                      test_data=test_data)
        sess.run(tf.initialize_all_variables())
        gpu_time = 0
        cpu_time = 0
        for repeat_id in range(100):
            random_ids = np.random.choice(
                gpu_policy_grads.validationDataForMDP.routingDataset.labelList.shape[0],
                250, replace=False)
            t0 = time.time()
            history_gpu = gpu_policy_grads. \
                sample_trajectories(sess=sess,
                                    routing_data=gpu_policy_grads.validationDataForMDP,
                                    state_sample_count=None,
                                    samples_per_state=1000,
                                    select_argmax=True,
                                    ignore_invalid_actions=ignore_invalid_actions,
                                    state_ids=random_ids)
            val_policy_value_gpu = gpu_policy_grads. \
                calculate_policy_value(sess=sess,
                                       routing_data=gpu_policy_grads.validationDataForMDP,
                                       state_batch_size=100,
                                       samples_per_state=10000)
            t1 = time.time()
            history_cpu = cpu_policy_grads.sample_trajectories(sess=sess,
                                                               routing_data=gpu_policy_grads.validationDataForMDP,
                                                               state_sample_count=None,
                                                               samples_per_state=1000,
                                                               select_argmax=True,
                                                               ignore_invalid_actions=ignore_invalid_actions,
                                                               state_ids=random_ids)
            val_policy_value_cpu = cpu_policy_grads. \
                calculate_policy_value(sess=sess,
                                       routing_data=gpu_policy_grads.validationDataForMDP,
                                       state_batch_size=100,
                                       samples_per_state=10000)
            print("val_policy_value_gpu={0}".format(val_policy_value_gpu))
            print("val_policy_value_cpu={0}".format(val_policy_value_cpu))
            t2 = time.time()
            gpu_time += (t1 - t0)
            cpu_time += (t2 - t1)
            assert np.array_equal(history_gpu.stateIds, history_cpu.stateIds)
            for t in range(network.depth - 1):
                assert np.allclose(history_gpu.states[t], history_cpu.states[t])
                if not ignore_invalid_actions:
                    assert np.allclose(history_gpu.policies[t], history_cpu.policies[t])
                else:
                    assert np.allclose(history_gpu.validPolicies[t], history_cpu.policies[t])
                assert np.array_equal(history_gpu.actions[t], history_cpu.actions[t])
                assert np.array_equal(history_gpu.rewards[t], history_cpu.rewards[t])
                assert np.array_equal(history_gpu.routingDecisions[t], history_cpu.routingDecisions[t])
            print("iteration:{0}".format(repeat_id))
        print("gpu_time={0}".format(gpu_time))
        print("cpu_time={0}".format(cpu_time))
        tf.reset_default_graph()


def train_policy_gradients_network():
    run_id = 453
    network_name = "FashionNet_Lite"
    iteration = 43680

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
    routing_data = network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                             output_names=output_names)
    validation_data, test_data = routing_data.apply_validation_test_split(test_ratio=0.1)

    wd_list = [0.0001]
    # [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005] * 10
    state_sample_count_list = [1000]
    samples_per_state_list = [2]
    ignore_invalid_actions = False
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[wd_list,
                                                                          state_sample_count_list,
                                                                          samples_per_state_list])

    for tpl in cartesian_product:
        sess = tf.Session()
        l2_wd = tpl[0]
        state_sample_count = tpl[1]
        samples_per_state = tpl[2]
        gpu_policy_grads = \
            FullGpuTreePolicyGradientsNetwork(l2_lambda=l2_wd,
                                              network=network,
                                              network_name=network_name,
                                              run_id=run_id,
                                              iteration=iteration,
                                              degree_list=[2, 2],
                                              output_names=output_names,
                                              used_feature_names=used_output_names,
                                              use_baselines=True,
                                              state_sample_count=state_sample_count,
                                              trajectory_per_state_sample_count=samples_per_state,
                                              hidden_layers=[[128], [256]],
                                              validation_data=validation_data,
                                              test_data=test_data,
                                              policy_network_func="mlp")
        gpu_policy_grads.train(sess=sess, max_num_of_iterations=10000)
        tf.reset_default_graph()
        # sess = tf.Session()
        # cpu_policy_grads = TreeDepthPolicyNetwork(l2_lambda=l2_wd,
        #                                           network=network,
        #                                           network_name=network_name,
        #                                           run_id=run_id,
        #                                           iteration=iteration,
        #                                           degree_list=[2, 2],
        #                                           output_names=output_names,
        #                                           used_feature_names=used_output_names,
        #                                           use_baselines=True,
        #                                           state_sample_count=state_sample_count,
        #                                           trajectory_per_state_sample_count=samples_per_state,
        #                                           hidden_layers=[[128], [256]],
        #                                           validation_data=validation_data,
        #                                           test_data=test_data)
        # cpu_policy_grads.train(sess=sess, max_num_of_iterations=20000)
        # tf.reset_default_graph()


def main():
    # compare_gpu_implementation()
    train_policy_gradients_network()


if __name__ == "__main__":
    main()
