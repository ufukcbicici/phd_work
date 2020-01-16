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
        self.actions = []
        self.rewards = []


class PolicyGradientsNetwork:
    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names, used_feature_names,
                 test_ratio=0.2):
        self.actionSpaces = None
        self.networkFeatureNames = output_names
        self.usedFeatureNames = used_feature_names
        self.l2Lambda = l2_lambda
        self.paramL2Norms = {}
        self.l2Loss = None
        self.inputs = []
        self.logits = []
        self.policies = []
        self.rewards = []
        self.validationFeaturesDict = {}
        self.testFeaturesDict = {}
        # Prepare CIGN topology data.
        self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        # Preparing training data for the policy gradients algorithm.
        self.routingData = self.network.load_routing_info(run_id=run_id, iteration=iteration, data_type=data_type,
                                                          output_names=self.networkFeatureNames)
        self.validationData, self.testData = self.routingData.apply_validation_test_split(test_ratio=test_ratio)
        self.validationMLPaths = self.get_max_likelihood_paths( branch_probs=self.validationData.get_dict("branch_probs"))
        self.testMLPaths = self.get_max_likelihood_paths(branch_probs=self.testData.get_dict("branch_probs"))
        self.validationFeaturesDict = self.prepare_state_features(data=self.validationData)
        self.testFeaturesDict = self.prepare_state_features(data=self.testData)
        self.build_action_spaces()

    # OK
    def prepare_state_features(self, data):
        pass

    # OK
    def build_action_spaces(self):
        pass

    # OK
    def build_networks(self):
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            self.build_state_inputs(time_step=t)
            self.build_policy_networks(time_step=t)

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
        self.inputs.append(state_input)

    # OK
    def build_policy_networks(self, time_step):
        pass

    # OK
    def build_rewards(self, time_step):
        action_count = self.actionSpaces[time_step]
        reward_input = tf.placeholder(dtype=tf.float32, shape=[None, action_count],
                                      name="rewards_{0}".format(time_step))
        self.rewards.append(reward_input)

    def sample_initial_states(self, data, features_dict, ml_selections_arr, state_sample_count, samples_per_state):
        pass

    def get_max_trajectory_length(self):
        pass

    def sample_from_policy(self, history, time_step):
        pass

    def state_transition(self, history):
        pass

    def prepare_sampling_feed_dict(self, curr_time_step):
        pass

    def build_policy_gradient_loss(self):
        pass

    def build_network(self):
        pass

    def train(self, state_sample_count, samples_per_state):
        pass
        # # State inputs and reward inputs
        # for t in range(self.trajectoryMaxLength):
        #     # States
        #     input_shape = [None]
        #     input_shape.extend(self.stateShapes[t])
        #     state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
        #     self.inputs.append(state_input)
        #     # Rewards
        #     reward_shape = [None, len(self.actionSpaces[t])]
        #     reward_input = tf.placeholder(dtype=tf.float32, shape=reward_shape, name="rewards_{0}".format(t))
        #     self.rewards.append(reward_input)
        # # Build policy generating networks; self.policies are filled.
        # self.build_policy_networks()

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
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

    def sample_trajectories(self, sess, data, features_dict, ml_selections_arr,
                            state_sample_count, samples_per_state):
        # Sample from s1 ~ p(s1)
        history = self.sample_initial_states(data=data,
                                             features_dict=features_dict,
                                             ml_selections_arr=ml_selections_arr,
                                             state_sample_count=state_sample_count,
                                             samples_per_state=samples_per_state)
        max_trajectory_length = self.get_max_trajectory_length()
        for t in range(max_trajectory_length):
            print("X")

