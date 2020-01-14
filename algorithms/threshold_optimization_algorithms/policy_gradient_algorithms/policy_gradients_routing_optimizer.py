import numpy as np
import tensorflow as tf
from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.combinatorial_routing_optimizer import CombinatorialRoutingOptimizer
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class TreeLevelRoutingOptimizer:
    def __init__(self, branching_state_vectors, hidden_layers, action_space_size):
        self.branchingStateVectors = branching_state_vectors
        # self.hiddenLayers = [self.branchingStateVectors.shape[-1]]
        # self.hiddenLayers.extend(hidden_layers)
        self.hiddenLayers = hidden_layers
        self.actionCount = int(action_space_size)
        self.inputs = None
        # Policy MLP
        self.net = None
        self.logits = None
        self.pi = None
        self.repeatedPolicy = None
        self.logRepeatedPolicy = None
        self.stateCount = None
        self.categoryCount = None
        # Policy Evaluation on the given data
        self.rewards = None
        self.weightedRewardMatrix = None
        self.valueFunctions = None
        self.policyValue = None
        self.policySampleCount = None
        self.policySamples = None
        self.logSampledPolicies = None
        self.rewardSamples = None
        self.proxyValueVector = None
        self.proxyPolicyValue = None
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.learningRate = tf.constant(0.00001)
        self.optimizer = None
        self.l2Loss = None
        self.totalLoss = None
        self.l2Lambda = PolicyGradientsRoutingOptimizer.L2_LAMBDA
        self.paramL2Norms = {}
        # Build network
        self.build_network()

    def build_network(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.branchingStateVectors.shape[-1]], name="inputs")
        self.policySampleCount = tf.placeholder(dtype=tf.int32, name="policySampleCount")
        self.hiddenLayers.append(self.actionCount)
        # Policy MLP
        self.net = self.inputs
        for layer_id, layer_dim in enumerate(self.hiddenLayers):
            if layer_id < len(self.hiddenLayers) - 1:
                self.net = tf.layers.dense(inputs=self.net, units=layer_dim, activation=tf.nn.relu)
            else:
                self.net = tf.layers.dense(inputs=self.net, units=layer_dim, activation=None)
        self.logits = self.net
        self.pi = tf.nn.softmax(self.logits)
        # Policy Evaluation on the given data
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, self.actionCount], name="rewards")
        self.weightedRewardMatrix = self.pi * self.rewards
        self.valueFunctions = tf.reduce_sum(self.weightedRewardMatrix, axis=1)
        self.policyValue = tf.reduce_mean(self.valueFunctions)
        # Proxy Objective Function and maximizing it
        self.repeatedPolicy = tf.tile(self.pi, [self.policySampleCount, 1])
        self.logRepeatedPolicy = tf.log(self.repeatedPolicy)
        self.policySamples = FastTreeNetwork.sample_from_categorical_v2(probs=self.repeatedPolicy)
        prob_shape = tf.shape(self.logRepeatedPolicy)
        self.stateCount = tf.gather_nd(prob_shape, [0])
        self.categoryCount = tf.gather_nd(prob_shape, [1])
        self.policySamples = tf.stack([tf.range(0, self.stateCount, 1), self.policySamples], axis=1)
        self.logSampledPolicies = tf.gather_nd(self.logRepeatedPolicy, self.policySamples)
        self.rewardSamples = tf.gather_nd(self.rewards, self.policySamples)
        self.proxyValueVector = self.logSampledPolicies * self.rewardSamples
        self.proxyPolicyValue = tf.reduce_mean(self.proxyValueVector)
        self.get_l2_loss()
        self.totalLoss = (-1.0 * self.proxyPolicyValue) + self.l2Loss
        # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate). \
            minimize(self.totalLoss, global_step=self.globalStep)

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
            self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)


class PolicyGradientsRoutingOptimizer(CombinatorialRoutingOptimizer):
    IMPOSSIBLE_ACTION_PENALTY = -1000.0
    CORRECT_PREDICTION_REWARD = 1.0
    INCORRECT_PREDICTION_REWARD = -1000.0
    MAC_PENALTY_COEFFICIENT = 0.0
    BATCH_SIZE = 10000
    TOTAL_ITERATIONS = 1000000
    L2_LAMBDA = 0.0

    def __init__(self, network_name, run_id, iteration, degree_list, data_type, output_names, test_ratio, features_used,
                 hidden_layers):
        super().__init__(network_name, run_id, iteration, degree_list, data_type, output_names)
        self.testRatio = test_ratio
        self.featuresUsed = features_used
        self.hiddenLayers = hidden_layers
        self.actionSpaces = [self.get_action_space(tree_level=tree_level)
                             for tree_level in range(self.network.depth - 1)]
        # Apply Validation - Test split to the routing data
        self.validationData, self.testData = self.routingData.apply_validation_test_split(test_ratio=self.testRatio)
        # Greedy Information Gain Paths
        self.validationDataPaths = self.get_max_likelihood_paths(
            branch_probs=self.validationData.get_dict("branch_probs"))
        self.testDataPaths = self.get_max_likelihood_paths(branch_probs=self.testData.get_dict("branch_probs"))
        # Enumerate possible validation and test states
        self.validationStateFeatures, self.validationNodeSelections = self.prepare_features_for_dataset(
            routing_dataset=self.validationData,
            greedy_routes=self.validationDataPaths)
        self.testStateFeatures, self.testNodeSelections = self.prepare_features_for_dataset(
            routing_dataset=self.testData,
            greedy_routes=self.testDataPaths)
        # self.test_route_size_compatibility(sample_routes=self.validationSampleRoutes)
        # self.test_route_size_compatibility(sample_routes=self.testSampleRoutes)
        # Enumerate rewards
        self.validationRewards = self.reward_function(states=self.validationStateFeatures,
                                                      labels=self.validationData.labelList,
                                                      node_selections_per_level=self.validationNodeSelections,
                                                      max_likelihood_routes=self.validationDataPaths,
                                                      posteriors=self.validationData.get_dict("posterior_probs"))
        self.testRewards = self.reward_function(states=self.testStateFeatures,
                                                labels=self.testData.labelList,
                                                node_selections_per_level=self.testNodeSelections,
                                                max_likelihood_routes=self.testDataPaths,
                                                posteriors=self.testData.get_dict("posterior_probs"))
        self.validationMaxLikelihoodAccuracy = self.get_max_likelihood_accuracy(
            max_likelihood_paths=self.validationDataPaths,
            posteriors=self.validationData.get_dict("posterior_probs"), labels=self.validationData.labelList)
        self.testMaxLikelihoodAccuracy = self.get_max_likelihood_accuracy(
            max_likelihood_paths=self.testDataPaths,
            posteriors=self.testData.get_dict("posterior_probs"), labels=self.testData.labelList)
        print("validationMaxLikelihoodAccuracy={0}".format(self.validationMaxLikelihoodAccuracy))
        print("testMaxLikelihoodAccuracy={0}".format(self.testMaxLikelihoodAccuracy))
        # Build Policy Gradient Networks
        self.policyGradientOptimizers = []
        for tree_level in range(self.network.depth - 1):
            action_space_size = len(self.actionSpaces[tree_level])
            policy_gradient_optimizer = TreeLevelRoutingOptimizer(
                branching_state_vectors=self.validationStateFeatures[tree_level],
                hidden_layers=hidden_layers[tree_level], action_space_size=action_space_size)
            self.policyGradientOptimizers.append(policy_gradient_optimizer)

    def test_route_size_compatibility(self, sample_routes):
        for idx in range(len(sample_routes)):
            assert all([
                np.array_equal(sample_routes[0][int(i / (sample_routes[idx].shape[0] / sample_routes[0].shape[0]))],
                               sample_routes[idx][i]) for i in range(sample_routes[idx].shape[0])])

    def get_action_space(self, tree_level):
        next_level_node_count = len([node for node in
                                     self.network.topologicalSortedNodes if node.depth == tree_level + 1])
        action_space_size = 2 ** next_level_node_count
        action_space = {}
        for action_id in range(action_space_size):
            l = [int(x) for x in list('{0:0b}'.format(action_id))]
            k = [0] * (next_level_node_count - len(l))
            k.extend(l)
            binary_node_selection = tuple(k)
            action_space[action_id] = binary_node_selection
        return action_space

    def get_reachability_dict(self, tree_level):
        next_level_valid_actions_dict = {}
        curr_level_nodes = self.network.orderedNodesPerLevel[tree_level]
        next_level_nodes = self.network.orderedNodesPerLevel[tree_level + 1]
        curr_level_route_combinations = UtilityFuncs.get_cartesian_product(
            list_of_lists=[[0, 1]] * len(curr_level_nodes))
        curr_level_route_combinations = [route for route in curr_level_route_combinations if sum(route) > 0]
        next_level_route_combinations = UtilityFuncs.get_cartesian_product(
            list_of_lists=[[0, 1]] * len(next_level_nodes))
        for route in curr_level_route_combinations:
            reachable_next_level_node_ids = set()
            parent_nodes = [node for i, node in enumerate(curr_level_nodes) if route[i] != 0]
            next_level_valid_actions_dict[route] = set()
            for parent_node in parent_nodes:
                child_nodes = {c_node.index for c_node in self.network.dagObject.children(node=parent_node)}
                reachable_next_level_node_ids = reachable_next_level_node_ids.union(child_nodes)
            for next_level_route in next_level_route_combinations:
                reached_nodes = {node.index for is_reached, node in zip(next_level_route, next_level_nodes)
                                 if is_reached != 0}
                if len(reached_nodes.difference(reachable_next_level_node_ids)) == 0:
                    next_level_valid_actions_dict[route].add(next_level_route)
        return next_level_valid_actions_dict

    def reward_function(self, states, labels, node_selections_per_level, max_likelihood_routes, posteriors):
        rewards_dict = {}
        posteriors_tensor = np.stack([posteriors[node.index] for node in self.leafNodes], axis=2)
        for tree_level in range(self.network.depth - 1):
            if tree_level != self.network.depth - 2:
                continue
            level_multiplicity = node_selections_per_level[tree_level].shape[0] / node_selections_per_level[0].shape[0]
            rewards = []
            next_level_valid_actions_dict = self.get_reachability_dict(tree_level=tree_level)
            next_level_min_id = min([n.index for n in self.network.orderedNodesPerLevel[tree_level + 1]])
            action_space = self.actionSpaces[tree_level]
            for idx in range(states[tree_level].shape[0]):
                sample_rewards = []
                true_label = labels[int(idx / level_multiplicity)]
                posteriors_matrix = posteriors_tensor[int(idx / level_multiplicity)]
                max_likelihood_route = max_likelihood_routes[int(idx / level_multiplicity)]
                curr_level_selected_nodes = node_selections_per_level[tree_level][idx]
                state = states[tree_level][idx]
                valid_actions = next_level_valid_actions_dict[tuple(curr_level_selected_nodes.tolist())]
                # Corresponds to binary mapping of each integer.
                for action_id, node_selection in action_space.items():
                    # Punish impossible actions
                    reward = 0.0
                    if node_selection not in valid_actions:
                        reward = PolicyGradientsRoutingOptimizer.IMPOSSIBLE_ACTION_PENALTY
                    # For a possible action: Correct,Incorrect Prediction Reward - MAC Cost
                    else:
                        # Check if class is correctly predicted
                        node_selection_with_max_likelihood = list(node_selection)
                        node_selection_with_max_likelihood[max_likelihood_route[tree_level + 1] - next_level_min_id] = 1
                        node_selection_with_max_likelihood = tuple(node_selection_with_max_likelihood)
                        uniform_weight = 1.0 / sum(node_selection_with_max_likelihood)
                        posteriors_sparse = posteriors_matrix * \
                                            (uniform_weight * np.expand_dims(
                                                np.array(node_selection_with_max_likelihood), axis=0))
                        posteriors_weighted = np.sum(posteriors_sparse, axis=1)
                        predicted_label = np.argmax(posteriors_weighted)
                        prediction_reward = PolicyGradientsRoutingOptimizer.CORRECT_PREDICTION_REWARD \
                            if predicted_label == true_label else \
                            PolicyGradientsRoutingOptimizer.INCORRECT_PREDICTION_REWARD
                        reward += prediction_reward
                        # Get the calculation cost
                        mac_cost = self.networkActivationCosts[node_selection_with_max_likelihood]
                        mac_cost = (mac_cost / self.baseEvaluationCost) - 1.0
                        reward -= PolicyGradientsRoutingOptimizer.MAC_PENALTY_COEFFICIENT * mac_cost
                    sample_rewards.append(reward)
                rewards.append(np.array(sample_rewards))
            rewards_dict[tree_level] = np.stack(rewards, axis=0)
        return rewards_dict

    def prepare_features_for_dataset(self, routing_dataset, greedy_routes):
        # Prepare Policy Gradients State Data
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        state_vectors_for_each_tree_level = []
        routes_per_sample = []
        route_combination_count = []
        for tree_level in range(self.network.depth - 1):
            state_vectors_for_each_tree_level.append([])
            routes_per_sample.append([])
            route_combination_count.append([])
        for idx in range(routing_dataset.labelList.shape[0]):
            route_arr = greedy_routes[idx]
            for tree_level in range(self.network.depth - 1):
                # Gather all feature dicts
                level_nodes = self.network.orderedNodesPerLevel[tree_level]
                route_combinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(level_nodes))
                route_combinations = [route for route in route_combinations if sum(route) > 0]
                min_level_id = min([node.index for node in level_nodes])
                selected_node_id = route_arr[tree_level]
                valid_node_selections = set()
                for route in route_combinations:
                    r = np.array(route)
                    r[selected_node_id - min_level_id] = 1
                    valid_node_selections.add(tuple(r))
                route_combination_count[tree_level].append(len(valid_node_selections))
                for route_combination in valid_node_selections:
                    level_features_list = []
                    for feature_name in self.featuresUsed:
                        feature_vectors_per_node = [routing_dataset.get_dict(feature_name)[node.index][idx]
                                                    for node in level_nodes]
                        weighted_vectors = [route_weight * f_vec for route_weight, f_vec in
                                            zip(route_combination, feature_vectors_per_node)]
                        feature_vector = np.concatenate(weighted_vectors, axis=-1)
                        level_features_list.append(feature_vector)
                    state_vector_for_curr_level = np.concatenate(level_features_list, axis=-1)
                    state_vectors_for_each_tree_level[tree_level].append(state_vector_for_curr_level)
                    routes_per_sample[tree_level].append(route_combination)
        for arr in route_combination_count:
            assert len(set(arr)) == 1
        for tree_level in range(len(state_vectors_for_each_tree_level)):
            state_vectors_for_each_tree_level[tree_level] = np.stack(state_vectors_for_each_tree_level[tree_level],
                                                                     axis=0)
            routes_per_sample[tree_level] = np.stack(routes_per_sample[tree_level], axis=0)
        return state_vectors_for_each_tree_level, routes_per_sample

    def sample_trajectory(self, sess, policy_gradient_network, **kwargs):
        tree_level = kwargs["tree_level"]
        state_sample_size = kwargs["state_sample_size"]
        policy_sample_size = kwargs["policy_sample_size"]
        states = kwargs["states"]
        rewards = kwargs["rewards"]
        # First, sample from states
        total_data_count = states.shape[0]
        sample_indices = np.random.choice(total_data_count, state_sample_size, replace=False)
        sampled_states = states[sample_indices]
        feed_dict = {policy_gradient_network.inputs: sampled_states,
                     policy_gradient_network.rewards: rewards,
                     policy_gradient_network.policySampleCount: policy_sample_size}
        # results = sess.run([policy_gradient_network.pi,
        #                     policy_gradient_network.logits,
        #                     policy_gradient_network.repeatedPolicy,
        #                     policy_gradient_network.policySamples,
        #                     policy_gradient_network.logRepeatedPolicy,
        #                     policy_gradient_network.stateCount,
        #                     policy_gradient_network.categoryCount,
        #                     policy_gradient_network.logSampledPolicies,
        #                     policy_gradient_network.rewardSamples,
        #                     policy_gradient_network.proxyValueVector,
        #                     policy_gradient_network.proxyPolicyValue],
        #                    feed_dict={policy_gradient_network.inputs: sampled_states,
        #                               policy_gradient_network.rewards: rewards,
        #                               policy_gradient_network.policySampleCount: policy_sample_size})
        results = sess.run([policy_gradient_network.optimizer,
                            policy_gradient_network.pi,
                            policy_gradient_network.proxyPolicyValue,
                            policy_gradient_network.rewards,
                            policy_gradient_network.rewardSamples],
                           feed_dict=feed_dict)
        print("Proxy value:{0}".format(results[-1]))

    def evaluate_policy(self, sess, policy_gradient_network, **kwargs):
        tree_level = kwargs["tree_level"]
        states = kwargs["states"]
        rewards = kwargs["rewards"]
        node_selections_per_level = kwargs["node_selections_per_level"]
        max_likelihood_routes = kwargs["max_likelihood_routes"]
        posteriors = kwargs["posteriors"]
        labels = kwargs["labels"]
        level_multiplicity = int(node_selections_per_level[tree_level].shape[0] / node_selections_per_level[0].shape[0])
        next_level_min_id = min([n.index for n in self.network.orderedNodesPerLevel[tree_level + 1]])
        posteriors_tensor = np.stack([posteriors[node.index] for node in self.leafNodes], axis=2)
        if tree_level != self.network.depth - 2:
            raise NotImplementedError()

        results = sess.run([policy_gradient_network.pi,
                            policy_gradient_network.weightedRewardMatrix,
                            policy_gradient_network.valueFunctions,
                            policy_gradient_network.policyValue],
                           feed_dict={policy_gradient_network.inputs: states,
                                      policy_gradient_network.rewards: rewards})
        policy = results[0]
        policy_value = results[-1]
        print("Policy Value:{0}".format(policy_value))
        all_cases_correct_count = 0
        sample_wise_predictions_dict = {}
        for idx in range(policy.shape[0]):
            sample_id = int(idx / level_multiplicity)
            max_likelihood_route = max_likelihood_routes[sample_id]
            posteriors_matrix = posteriors_tensor[sample_id]
            true_label = labels[sample_id]
            best_case_correct = False
            if sample_id not in sample_wise_predictions_dict:
                sample_wise_predictions_dict[sample_id] = []
            state_policy = policy[idx]
            selected_action = int(np.argmax(state_policy))
            node_selection = self.actionSpaces[tree_level][selected_action]
            node_selection_with_max_likelihood = list(node_selection)
            node_selection_with_max_likelihood[max_likelihood_route[tree_level + 1] - next_level_min_id] = 1
            node_selection_with_max_likelihood = tuple(node_selection_with_max_likelihood)
            uniform_weight = 1.0 / sum(node_selection_with_max_likelihood)
            posteriors_sparse = posteriors_matrix * \
                                (uniform_weight * np.expand_dims(
                                    np.array(node_selection_with_max_likelihood), axis=0))
            posteriors_weighted = np.sum(posteriors_sparse, axis=1)
            predicted_label = np.argmax(posteriors_weighted)
            is_correct = predicted_label == true_label
            sample_wise_predictions_dict[sample_id].append(is_correct)
            all_cases_correct_count += float(is_correct)
        all_cases_accuracy = all_cases_correct_count / policy.shape[0]
        sample_wise_predictions = [any(pred_list) for pred_list in sample_wise_predictions_dict.values()]
        best_cases_accuracy = sum(sample_wise_predictions) / float(len(sample_wise_predictions))
        print("all_cases_accuracy={0}".format(all_cases_accuracy))
        print("best_cases_accuracy={0}".format(best_cases_accuracy))

    def train(self):
        for tree_level in range(self.network.depth - 2, -1, -1):
            if tree_level != self.network.depth - 2:
                continue
            # Init network
            policy_gradient_network = self.policyGradientOptimizers[tree_level]
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            print("Before Start Validation Value")
            self.evaluate_policy(sess=sess,
                                 policy_gradient_network=policy_gradient_network,
                                 tree_level=tree_level,
                                 states=self.validationStateFeatures[tree_level],
                                 rewards=self.validationRewards[tree_level],
                                 labels=self.validationData.labelList,
                                 node_selections_per_level=self.validationNodeSelections,
                                 max_likelihood_routes=self.validationDataPaths,
                                 posteriors=self.validationData.get_dict("posterior_probs")
                                 )
            print("Before Start Test Value")
            self.evaluate_policy(sess=sess,
                                 policy_gradient_network=policy_gradient_network,
                                 tree_level=tree_level,
                                 states=self.testStateFeatures[tree_level],
                                 rewards=self.testRewards[tree_level],
                                 labels=self.testData.labelList,
                                 node_selections_per_level=self.testNodeSelections,
                                 max_likelihood_routes=self.testDataPaths,
                                 posteriors=self.testData.get_dict("posterior_probs")
                                 )
            # Train with Policy Gradients
            for iteration_id in range(PolicyGradientsRoutingOptimizer.TOTAL_ITERATIONS):
                self.sample_trajectory(sess=sess,
                                       policy_gradient_network=policy_gradient_network,
                                       tree_level=tree_level,
                                       state_sample_size=self.validationStateFeatures[tree_level].shape[0],
                                       policy_sample_size=100,
                                       states=self.validationStateFeatures[tree_level],
                                       rewards=self.validationRewards[tree_level])
                if iteration_id % 10 == 0:
                    print("**********************************")
                    print("Iteration {0} validation policy Value:".format(iteration_id))
                    self.evaluate_policy(sess=sess,
                                         policy_gradient_network=policy_gradient_network,
                                         tree_level=tree_level,
                                         states=self.validationStateFeatures[tree_level],
                                         rewards=self.validationRewards[tree_level],
                                         labels=self.validationData.labelList,
                                         node_selections_per_level=self.validationNodeSelections,
                                         max_likelihood_routes=self.validationDataPaths,
                                         posteriors=self.validationData.get_dict("posterior_probs")
                                         )
                    print("Iteration {0} test policy Value:".format(iteration_id))
                    self.evaluate_policy(sess=sess,
                                         policy_gradient_network=policy_gradient_network,
                                         tree_level=tree_level,
                                         states=self.testStateFeatures[tree_level],
                                         rewards=self.testRewards[tree_level],
                                         labels=self.testData.labelList,
                                         node_selections_per_level=self.testNodeSelections,
                                         max_likelihood_routes=self.testDataPaths,
                                         posteriors=self.testData.get_dict("posterior_probs")
                                         )
                    print("**********************************")

def main():
    # run_id = 715
    # network_name = "Cifar100_CIGN_MultiGpuSingleLateExit"
    # iteration = 119100

    run_id = 453
    network_name = "FashionNet_Lite"
    iteration = 43680

    output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                    "pre_branch_feature"]
    policy_gradients_routing_optimizer = PolicyGradientsRoutingOptimizer(network_name=network_name, run_id=run_id,
                                                                         iteration=iteration,
                                                                         degree_list=[2, 2], data_type="test",
                                                                         output_names=output_names,
                                                                         test_ratio=0.2,
                                                                         features_used=["branching_feature"],
                                                                         hidden_layers=[[256], [512]])
    policy_gradients_routing_optimizer.train()


if __name__ == "__main__":
    main()