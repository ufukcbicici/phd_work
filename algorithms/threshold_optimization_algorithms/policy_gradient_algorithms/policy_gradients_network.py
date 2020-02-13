import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


# class MarkovDecisionProcessTimeStep:
#     def __init__(self, s, a=None, r=None):
#         self.state = s
#         self.action = a
#         self.reward = r

class TrajectoryHistory:
    def __init__(self, state_ids, max_likelihood_routes):
        self.stateIds = state_ids
        self.maxLikelihoodRoutes = max_likelihood_routes
        self.states = []
        self.stateIdsTemporal = []
        self.policies = []
        self.validPolicies = []
        self.actions = []
        self.routingDecisions = []
        self.rewards = []


class RoutingDataForMDP:
    def __init__(self, routing_dataset, features_dict, ml_paths, posteriors_tensor):
        self.routingDataset = routing_dataset
        self.featuresDict = features_dict
        self.mlPaths = ml_paths
        self.posteriorsTensor = posteriors_tensor
        self.rewardTensors = None


class TrajectoryAccuracy:
    def __init__(self):
        self.accuracyAllActions = None
        self.accuracyValidActions = None
        self.accuracyCombinedWithIG = None
        self.computationLoad = None
        self.computationLoadWithIG = None


class PolicyGradientsNetwork:
    def __init__(self, validation_data, test_data, l2_lambda,
                 network, network_name, run_id, iteration, degree_list, output_names, used_feature_names,
                 use_baselines, state_sample_count, trajectory_per_state_sample_count):
        self.actionSpaces = None
        self.networkFeatureNames = output_names
        self.usedFeatureNames = used_feature_names
        self.l2Lambda = l2_lambda
        self.l2LambdaTf = tf.placeholder(dtype=tf.float32, name="l2LambdaTf")
        self.paramL2Norms = {}
        self.l2Loss = None
        self.stateInputs = []
        self.logits = []
        self.policies = []
        self.logPolicies = []
        self.policySamples = []
        self.policyArgMaxSamples = []
        self.selectedPolicyInputs = []
        self.selectedLogPolicySamples = []
        self.proxyLossTrajectories = []
        self.proxyLossVector = None
        self.proxyLoss = None
        self.totalLoss = None
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.learningRate = None
        self.trajectoryCount = None
        self.rewards = []
        self.cumulativeRewards = []
        self.baselinesTf = []
        self.baselinesNp = []
        self.weightedRewardMatrices = []
        self.valueFunctions = None
        self.validationFeaturesDict = {}
        self.testFeaturesDict = {}
        self.networkActivationCosts = None
        self.networkActivationCostsDict = {}
        self.baseEvaluationCost = None
        self.reachabilityMatrices = []
        self.optimizer = None
        self.stateSampleCount = state_sample_count
        self.trajectoryPerStateSampleCount = trajectory_per_state_sample_count
        self.useBaselines = use_baselines
        self.networkName = network_name
        self.networkRunId = run_id
        self.networkIteration = iteration
        # Prepare CIGN topology data.
        self.network = network
        # self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        # Preparing training data for the policy gradients algorithm.
        # self.routingData = self.network.load_routing_info(run_id=run_id, iteration=iteration, data_type=data_type,
        #                                                   output_names=self.networkFeatureNames)
        # self.validationData, self.testData = self.routingData.apply_validation_test_split(test_ratio=test_ratio)
        self.validationData = validation_data
        self.testData = test_data
        self.validationMLPaths = self.get_max_likelihood_paths(
            branch_probs=self.validationData.get_dict("branch_probs"))
        self.testMLPaths = self.get_max_likelihood_paths(branch_probs=self.testData.get_dict("branch_probs"))
        self.validationFeaturesDict = self.prepare_state_features(data=self.validationData)
        self.testFeaturesDict = self.prepare_state_features(data=self.testData)
        self.validationPosteriorsTensor = \
            np.stack([self.validationData.get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)
        self.testPosteriorsTensor = \
            np.stack([self.testData.get_dict("posterior_probs")[node.index] for node in self.leafNodes], axis=2)
        self.validationDataForMDP = RoutingDataForMDP(
            routing_dataset=self.validationData,
            features_dict=self.validationFeaturesDict,
            ml_paths=self.validationMLPaths,
            posteriors_tensor=self.validationPosteriorsTensor)
        self.testDataForMDP = RoutingDataForMDP(
            routing_dataset=self.testData,
            features_dict=self.testFeaturesDict,
            ml_paths=self.testMLPaths,
            posteriors_tensor=self.testPosteriorsTensor)
        self.build_action_spaces()
        self.get_evaluation_costs()
        self.get_reachability_matrices()
        self.build_networks()

    # OK
    def prepare_state_features(self, data):
        pass

    # OK
    def build_action_spaces(self):
        pass

    # OK
    def get_reachability_matrices(self):
        pass

    # OK
    def build_networks(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            self.build_state_inputs(time_step=t)
            self.build_policy_networks(time_step=t)
            reward_input = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards_{0}".format(t))
            self.rewards.append(reward_input)
            baseline_input = tf.placeholder(dtype=tf.float32, shape=[None], name="baselines_{0}".format(t))
            self.baselinesTf.append(baseline_input)
            if t - 1 < 0:
                baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0], 1))
            else:
                baseline_np = np.zeros(shape=(self.validationDataForMDP.routingDataset.labelList.shape[0],
                                              self.actionSpaces[t - 1].shape[0]))
            self.baselinesNp.append(baseline_np)
            # Policy sampling
            sampler = FastTreeNetwork.sample_from_categorical_v2(probs=self.policies[t])
            self.policySamples.append(sampler)
            selected_policy_input = tf.placeholder(dtype=tf.int32, shape=[None],
                                                   name="selected_policy_input_{0}".format(t))
            self.selectedPolicyInputs.append(selected_policy_input)
            # Argmax actions
            argmax_actions = tf.argmax(self.policies[t], axis=1)
            self.policyArgMaxSamples.append(argmax_actions)
        # Get the total number of trajectories
        state_input_shape = tf.shape(self.stateInputs[0])
        self.trajectoryCount = tf.gather_nd(state_input_shape, [0])
        # Cumulative Rewards
        for t1 in range(max_trajectory_length):
            rew_list = [self.rewards[t2] for t2 in range(t1, max_trajectory_length, 1)]
            cum_sum = tf.add_n(rew_list)
            self.cumulativeRewards.append(cum_sum)
        # Building the proxy loss and the policy gradient
        self.build_policy_gradient_loss()
        self.get_l2_loss()
        self.build_optimizer()

    # OK
    def build_state_inputs(self, time_step):
        ordered_nodes_at_level = self.network.orderedNodesPerLevel[time_step]
        inputs_list = [self.validationFeaturesDict[node.index] for node in ordered_nodes_at_level]
        shape_set = {input_arr.shape for input_arr in inputs_list}
        assert len(shape_set) == 1
        concated_feat = np.concatenate(inputs_list, axis=-1)
        input_shape = [None]
        input_shape.extend(concated_feat.shape[1:])
        input_shape = tuple(input_shape)
        state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(time_step))
        self.stateInputs.append(state_input)

    # OK
    def build_policy_networks(self, time_step):
        pass

    def build_policy_gradient_loss(self):
        pass

    def build_optimizer(self):
        self.learningRate = tf.constant(0.0001)
        self.totalLoss = (-1.0 * self.proxyLoss) + self.l2Loss
        # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
        boundaries = [10000]
        values = [0.0001, 0.00001]
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(
            self.totalLoss, global_step=self.globalStep)

        # self.learningRate = tf.train.piecewise_constant(self.globalStep, boundaries, values)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(
        #     self.totalLoss, global_step=self.globalStep)
        # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(
        #     self.totalLoss, global_step=self.globalStep)

    # OK
    def sample_initial_states(self, routing_data, state_sample_count, samples_per_state,
                              state_ids=None) -> TrajectoryHistory:
        pass

    # OK
    def get_max_trajectory_length(self) -> int:
        pass

    # OK
    def sample_from_policy(self, sess, routing_data, history, time_step, select_argmax, ignore_invalid_actions):
        pass

    # OK
    def state_transition(self, routing_data, history, time_step):
        pass

    # OK
    def reward_calculation(self, routing_data, history, time_step):
        pass

    # OK
    def sample_trajectories(self, sess, routing_data, state_sample_count, samples_per_state,
                            select_argmax, ignore_invalid_actions, state_ids) \
            -> TrajectoryHistory:
        # if state_ids is None, sample from state distribution
        # Sample from s1 ~ p(s1)
        history = self.sample_initial_states(routing_data=routing_data,
                                             state_sample_count=state_sample_count,
                                             samples_per_state=samples_per_state,
                                             state_ids=state_ids)
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            # Sample from a_t ~ p(a_t|history(t))
            self.sample_from_policy(sess=sess, routing_data=routing_data, history=history, time_step=t,
                                    select_argmax=select_argmax, ignore_invalid_actions=ignore_invalid_actions)
            # Get the reward: r_t ~ p(r_t|history(t))
            self.reward_calculation(routing_data=routing_data, history=history, time_step=t)
            # State transition s_{t+1} ~ p(s_{t+1}|history(t))
            if t < max_trajectory_length - 1:
                self.state_transition(routing_data=routing_data, history=history, time_step=t)
        return history

    def update_baselines(self, history):
        pass

    # def calculate_accuracy_of_trajectories_with_ig(self, routing_data, history):
    #     min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
    #     routing_decisions = np.copy(history.routingDecisions[-1])
    #     leaf_ids = routing_data.mlPaths[history.stateIds, -1] - min_leaf_id
    #     routing_decisions[:, leaf_ids] = 1.0
    #     validity_of_predictions_vec = self.calculate_accuracy_of_trajectories(routing_data=routing_data,
    #                                                                           history=history,
    #                                                                           routing_decisions=routing_decisions)
    #     return validity_of_predictions_vec

    def calculate_accuracy_of_trajectories(self, routing_data, history, combine_with_ig):
        true_labels = routing_data.routingDataset.labelList[history.stateIds]
        posteriors = routing_data.posteriorsTensor[history.stateIds, :]
        routing_decisions = np.copy(history.routingDecisions[-1])
        assert routing_decisions.shape[1] == posteriors.shape[2]
        if combine_with_ig:
            min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
            routing_decisions = np.copy(history.routingDecisions[-1])
            leaf_ids = routing_data.mlPaths[history.stateIds, -1] - min_leaf_id
            routing_decisions[np.arange(routing_decisions.shape[0]), leaf_ids] = 1.0
        routing_weights = np.reciprocal(np.sum(routing_decisions.astype(np.float32), axis=1))
        routing_decisions_t_weighted = routing_decisions * np.expand_dims(routing_weights, axis=1)
        weighted_posteriors = posteriors * np.expand_dims(routing_decisions_t_weighted, axis=1)
        final_posteriors = np.sum(weighted_posteriors, axis=2)
        predicted_labels = np.argmax(final_posteriors, axis=1)
        validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
        # Calculate the computation overload
        computation_overload_vector = np.apply_along_axis(lambda x: self.networkActivationCostsDict[tuple(x)], axis=1,
                                                          arr=routing_decisions)
        return validity_of_predictions_vec, computation_overload_vector

    # OK
    # def calculate_accuracy_of_trajectories(self, routing_data, history):
    #     true_labels = routing_data.routingDataset.labelList[history.stateIds]
    #     posteriors = routing_data.posteriorsTensor[history.stateIds, :]
    #     routing_decisions_t = history.routingDecisions[-1]
    #     assert routing_decisions_t.shape[1] == posteriors.shape[2]
    #     routing_weights = np.reciprocal(np.sum(routing_decisions_t.astype(np.float32), axis=1))
    #     routing_decisions_t_weighted = routing_decisions_t * np.expand_dims(routing_weights, axis=1)
    #     weighted_posteriors = posteriors * np.expand_dims(routing_decisions_t_weighted, axis=1)
    #     final_posteriors = np.sum(weighted_posteriors, axis=2)
    #     predicted_labels = np.argmax(final_posteriors, axis=1)
    #     validity_of_predictions_vec = (predicted_labels == true_labels).astype(np.int32)
    #     return validity_of_predictions_vec

    # OK
    def calculate_policy_value(self, sess, routing_data, state_batch_size, samples_per_state):
        # self, data, features_dict, ml_selections_arr, posteriors_tensor,
        # state_sample_count, samples_per_state, state_ids = None
        # curr_state_id = 0
        total_rewards = 0.0
        trajectory_count = 0.0
        data_count = routing_data.routingDataset.labelList.shape[0]
        id_list = list(range(data_count))
        for idx in range(0, data_count, state_batch_size):
            curr_sample_ids = id_list[idx:idx + state_batch_size]
            history = self.sample_trajectories(routing_data=routing_data,
                                               state_sample_count=None,
                                               samples_per_state=samples_per_state,
                                               select_argmax=False,
                                               ignore_invalid_actions=False,
                                               state_ids=curr_sample_ids,
                                               sess=sess)
            rewards_matrix = np.stack([history.rewards[t] for t in range(self.get_max_trajectory_length())], axis=1)
            total_rewards += np.sum(rewards_matrix)
            trajectory_count += rewards_matrix.shape[0]
        expected_policy_value = total_rewards / trajectory_count
        return expected_policy_value

    # OK
    def calculate_routing_accuracy(self, sess, routing_data, state_batch_size):
        pass

    def get_max_likelihood_accuracy(self, routing_data):
        sample_counts_set = set()
        sample_counts_set.add(routing_data.posteriorsTensor.shape[0])
        sample_counts_set.add(routing_data.mlPaths.shape[0])
        sample_counts_set.add(routing_data.routingDataset.labelList.shape[0])
        assert len(sample_counts_set) == 1
        correct_count = 0
        min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
        for idx in range(routing_data.routingDataset.labelList.shape[0]):
            leaf_id = routing_data.mlPaths[idx, -1]
            posterior = routing_data.posteriorsTensor[idx, :, leaf_id - min_leaf_id]
            predicted_label = np.argmax(posterior)
            correct_count += float(predicted_label == routing_data.routingDataset.labelList[idx])
        accuracy = correct_count / routing_data.routingDataset.labelList.shape[0]
        return accuracy

    def get_max_likelihood_accuracy_static(self, routing_data, ml_paths, posteriors):
        sample_counts_set = set()
        sample_counts_set.add(routing_data.posteriorsTensor.shape[0])
        sample_counts_set.add(routing_data.mlPaths.shape[0])
        sample_counts_set.add(routing_data.routingDataset.labelList.shape[0])
        assert len(sample_counts_set) == 1
        correct_count = 0
        min_leaf_id = min([node.index for node in self.network.orderedNodesPerLevel[self.network.depth - 1]])
        for idx in range(routing_data.routingDataset.labelList.shape[0]):
            leaf_id = routing_data.mlPaths[idx, -1]
            posterior = routing_data.posteriorsTensor[idx, :, leaf_id - min_leaf_id]
            predicted_label = np.argmax(posterior)
            correct_count += float(predicted_label == routing_data.routingDataset.labelList[idx])
        accuracy = correct_count / routing_data.routingDataset.labelList.shape[0]
        return accuracy

    # OK
    def evaluate_policy_values(self, sess):
        validation_policy_value = self.calculate_policy_value(routing_data=self.validationDataForMDP,
                                                              state_batch_size=100, samples_per_state=10,
                                                              sess=sess)
        test_policy_value = self.calculate_policy_value(routing_data=self.testDataForMDP,
                                                        state_batch_size=100, samples_per_state=10, sess=sess)
        print("validation_policy_value={0}".format(validation_policy_value))
        print("test_policy_value={0}".format(test_policy_value))
        return validation_policy_value, test_policy_value

    def evaluate_routing_accuracies(self, sess):
        validation_accuracy = self.calculate_routing_accuracy(sess=sess,
                                                              routing_data=self.validationDataForMDP,
                                                              state_batch_size=1000)
        test_accuracy = self.calculate_routing_accuracy(sess=sess,
                                                        routing_data=self.testDataForMDP, state_batch_size=100)
        print("validation_accuracy={0}".format(validation_accuracy))
        print("test_accuracy={0}".format(test_accuracy))
        return validation_accuracy, test_accuracy

    def evaluate_ml_routing_accuracies(self):
        val_ml_accuracy = self.get_max_likelihood_accuracy(routing_data=self.validationDataForMDP)
        test_ml_accuracy = self.get_max_likelihood_accuracy(routing_data=self.testDataForMDP)
        print("val_ml_accuracy={0}".format(val_ml_accuracy))
        print("test_ml_accuracy={0}".format(test_ml_accuracy))
        return val_ml_accuracy, test_ml_accuracy

    def train(self, sess, max_num_of_iterations):
        pass
        # # State stateInputs and reward stateInputs
        # for t in range(self.trajectoryMaxLength):
        #     # States
        #     input_shape = [None]
        #     input_shape.extend(self.stateShapes[t])
        #     state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
        #     self.stateInputs.append(state_input)
        #     # Rewards
        #     reward_shape = [None, len(self.actionSpaces[t])]
        #     reward_input = tf.placeholder(dtype=tf.float32, shape=reward_shape, name="rewards_{0}".format(t))
        #     self.rewards.append(reward_input)
        # # Build policy generating networks; self.policies are filled.
        # self.build_policy_networks()

    def get_evaluation_costs(self):
        list_of_lists = []
        path_costs = []
        for node in self.leafNodes:
            list_of_lists.append([0, 1])
            leaf_ancestors = self.network.dagObject.ancestors(node=node)
            leaf_ancestors.append(node)
            path_costs.append(sum([self.network.nodeCosts[ancestor.index] for ancestor in leaf_ancestors]))
        self.baseEvaluationCost = np.mean(np.array(path_costs))
        self.networkActivationCosts = []
        for action_id in range(self.actionSpaces[-1].shape[0]):
            node_selection = self.actionSpaces[-1][action_id]
            processed_nodes_set = set()
            for node_idx, curr_node in enumerate(self.leafNodes):
                if node_selection[node_idx] == 0:
                    continue
                leaf_ancestors = self.network.dagObject.ancestors(node=curr_node)
                leaf_ancestors.append(curr_node)
                for ancestor in leaf_ancestors:
                    processed_nodes_set.add(ancestor.index)
            total_cost = sum([self.network.nodeCosts[n_idx] for n_idx in processed_nodes_set])
            self.networkActivationCosts.append(total_cost)
            self.networkActivationCostsDict[tuple(self.actionSpaces[-1][action_id])] = \
                total_cost / self.baseEvaluationCost
        self.networkActivationCosts = np.array(self.networkActivationCosts) * (1.0 / self.baseEvaluationCost)

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2LambdaTf * tf.nn.l2_loss(tv)
            self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)

    def get_max_likelihood_paths(self, branch_probs):
        sample_sizes = list(set([arr.shape[0] for arr in branch_probs.values()]))
        assert len(sample_sizes) == 1
        sample_size = sample_sizes[0]
        max_likelihood_paths = []
        for idx in range(sample_size):
            curr_node = self.network.topologicalSortedNodes[0]
            route = []
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    break
                routing_distribution = branch_probs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            max_likelihood_paths.append(np.array(route))
        max_likelihood_paths = np.stack(max_likelihood_paths, axis=0)
        return max_likelihood_paths

    def get_explanation(self):
        pass
