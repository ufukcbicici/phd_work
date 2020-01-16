import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    PolicyGradientsNetwork, TrajectoryHistory


class TreeDepthState:
    def __init__(self, state_ids, state_vecs, max_likelihood_selections):
        self.stateIds = state_ids
        self.stateVectors = state_vecs
        self.maxLikelihoodSelections = max_likelihood_selections


class TreeDepthPolicyNetwork(PolicyGradientsNetwork):
    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names, used_feature_names,
                 hidden_layers, test_ratio=0.2):
        self.hiddenLayers = hidden_layers
        self.logits = []
        assert len(self.hiddenLayers) == self.get_max_trajectory_length()
        super().__init__(l2_lambda, network_name, run_id, iteration, degree_list, data_type, output_names,
                         used_feature_names, test_ratio=test_ratio)

    def prepare_state_features(self, data):
        # Prepare Policy Gradients State Data
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        features_dict = {}
        for node in self.innerNodes:
            # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            array_list = []
            for feature_name in self.usedFeatureNames:
                feature_arr = data.get_dict(feature_name)[node.index]
                if len(feature_arr.shape) > 2:
                    shape_as_list = list(feature_arr.shape)
                    mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
                    feature_arr = np.mean(feature_arr, axis=mean_axes)
                array_list.append(feature_arr)
            feature_vectors = np.concatenate(array_list, axis=-1)
            features_dict[node.index] = feature_vectors
        return features_dict

    def build_action_spaces(self):
        max_trajectory_length = self.get_max_trajectory_length()
        self.actionSpaces = []
        for t in range(max_trajectory_length):
            next_level_node_count = len(self.network.orderedNodesPerLevel[t+1])
            action_count = (2 ** next_level_node_count) - 1
            action_space = {}
            for action_id in range(action_count):
                action_code = action_id + 1
                l = [int(x) for x in list('{0:0b}'.format(action_code))]
                k = [0] * (next_level_node_count - len(l))
                k.extend(l)
                binary_node_selection = tuple(k)
                action_space[action_id] = binary_node_selection
            self.actionSpaces.append(action_space)

    def build_policy_networks(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            # Create state input for the time step t
            input_shape = [None]
            ordered_nodes_at_level = self.network.orderedNodesPerLevel[t]
            inputs_list = [self.validationFeaturesDict[node.index] for node in ordered_nodes_at_level]
            shape_set = {input_arr.shape for input_arr in inputs_list}
            assert len(shape_set) == 1
            concated_feat = np.concatenate(inputs_list, axis=-1)
            input_shape.extend(concated_feat.shape[1:])
            input_shape = tuple(input_shape)
            state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
            self.inputs.append(state_input)
            # Create the policy network
            hidden_layers = list(self.hiddenLayers[t])
            hidden_layers.append(len(self.actionSpaces[t]))
            net = self.inputs[t]
            for layer_id, layer_dim in enumerate(hidden_layers):
                if layer_id < len(hidden_layers) - 1:
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
                else:
                    net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
            self.logits.append(net)
            self.policies.append(tf.nn.softmax(self.logits))

    def sample_initial_states(self, data, features_dict, ml_selections_arr, state_sample_count, samples_per_state):
        total_sample_count = data.labelList.shape[0]
        sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
        sample_indices = np.repeat(sample_indices, repeats=samples_per_state)
        feature_arr = features_dict[self.network.topologicalSortedNodes[0].index]
        initial_state_vectors = feature_arr[sample_indices, :]
        state_ml_selections = ml_selections_arr[sample_indices, :]
        history = TrajectoryHistory(state_ids=sample_indices, max_likelihood_routes=state_ml_selections)
        history.states.append(initial_state_vectors)
        return history

    def get_max_trajectory_length(self):
        return self.network.depth - 1

    def sample_from_policy(self, history, time_step):
        assert len(history.states) == time_step + 1

    def state_transition(self, history):
        pass

    def train(self, state_sample_count, samples_per_state):
        sess = tf.Session()
        self.sample_trajectories(sess=sess, data=self.validationData, features_dict=self.validationFeaturesDict,
                                 ml_selections_arr=self.validationMLPaths, state_sample_count=state_sample_count,
                                 samples_per_state=samples_per_state)
        print("X")


def main():
    # run_id = 715
    # network_name = "Cifar100_CIGN_MultiGpuSingleLateExit"
    # iteration = 119100

    run_id = 453
    network_name = "FashionNet_Lite"
    iteration = 43680

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    used_output_names = ["pre_branch_feature"]
    policy_gradients_routing_optimizer = TreeDepthPolicyNetwork(l2_lambda=0.0,
                                                                network_name=network_name,
                                                                run_id=run_id,
                                                                iteration=iteration,
                                                                degree_list=[2, 2],
                                                                data_type="test",
                                                                output_names=output_names,
                                                                used_feature_names=used_output_names,
                                                                test_ratio=0.2,
                                                                hidden_layers=[[128], [256]])
    state_sample_count = policy_gradients_routing_optimizer.validationData.labelList.shape[0]
    samples_per_state = 100
    policy_gradients_routing_optimizer.train(state_sample_count=state_sample_count, samples_per_state=samples_per_state)


if __name__ == "__main__":
    main()
    # def prepare_sampling_feed_dict(self, curr_time_step):
    #     feed_dict = {}
    #     for tau in range(curr_time_step):
    #         total_sample_count = data.labelList.shape[0]
    #         sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
    #         sample_indices = np.repeat(sample_indices, repeats=samples_per_state)

    # def prepare_state_features(self, data):
    #     # Prepare Policy Gradients State Data
    #     root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
    #     assert len(root_node) == 1
    #     for idx in range(data.labelList.shape[0]):
    #
    #     for tree_level in range(self.network.depth - 1):
    #         state_vectors_for_each_tree_level.append([])
    #         routes_per_sample.append([])
    #         route_combination_count.append([])
    #     for idx in range(routing_dataset.labelList.shape[0]):
    #         route_arr = greedy_routes[idx]
    #         for tree_level in range(self.network.depth - 1):
    #             # Gather all feature dicts
    #             level_nodes = self.network.orderedNodesPerLevel[tree_level]
    #             route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(level_nodes))
    #             route_combinations = [route for route in route_combinations if sum(route) > 0]
    #             min_level_id = min([node.index for node in level_nodes])
    #             selected_node_id = route_arr[tree_level]
    #             valid_node_selections = set()
    #             for route in route_combinations:
    #                 r = np.array(route)
    #                 r[selected_node_id - min_level_id] = 1
    #                 valid_node_selections.add(tuple(r))
    #             route_combination_count[tree_level].append(len(valid_node_selections))
    #             for route_combination in valid_node_selections:
    #                 level_features_list = []
    #                 for feature_name in self.featuresUsed:
    #                     feature_vectors_per_node = [routing_dataset.get_dict(feature_name)[node.index][idx]
    #                                                 for node in level_nodes]
    #                     weighted_vectors = [route_weight * f_vec for route_weight, f_vec in
    #                                         zip(route_combination, feature_vectors_per_node)]
    #                     feature_vector = np.concatenate(weighted_vectors, axis=-1)
    #                     level_features_list.append(feature_vector)
    #                 state_vector_for_curr_level = np.concatenate(level_features_list, axis=-1)
    #                 state_vectors_for_each_tree_level[tree_level].append(state_vector_for_curr_level)
    #                 routes_per_sample[tree_level].append(route_combination)
    #     for arr in route_combination_count:
    #         assert len(set(arr)) == 1
    #     for tree_level in range(len(state_vectors_for_each_tree_level)):
    #         state_vectors_for_each_tree_level[tree_level] = np.stack(state_vectors_for_each_tree_level[tree_level],
    #                                                                  axis=0)
    #         routes_per_sample[tree_level] = np.stack(routes_per_sample[tree_level], axis=0)
    #     return state_vectors_for_each_tree_level, routes_per_sample
