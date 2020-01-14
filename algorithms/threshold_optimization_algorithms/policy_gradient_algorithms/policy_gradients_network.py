import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class MarkovDecisionProcessTimeStep:
    def __init__(self, s, a=None, r=None):
        self.state = s
        self.action = a
        self.reward = r


class PolicyGradientsNetwork:
    def __init__(self, l2_lambda,
                 network_name, run_id, iteration, degree_list, data_type, output_names,
                 test_ratio=0.2):
        self.actionSpaces = None
        self.networkFeatureNames = output_names
        self.l2Lambda = l2_lambda
        self.paramL2Norms = {}
        self.l2Loss = None
        self.inputs = []
        self.policies = []
        self.rewards = []
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

    def build_policy_networks(self):
        pass

    def sample_initial_states(self, data, state_sample_count, samples_per_state):
        pass

    def state_transition(self, history):
        pass

    def prepare_sampling_feed_dict(self, curr_time_step):
        pass

    def build_policy_gradient_loss(self):
        pass

    def build_network(self):
        # State inputs and reward inputs
        for t in range(self.trajectoryMaxLength):
            # States
            input_shape = [None]
            input_shape.extend(self.stateShapes[t])
            state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
            self.inputs.append(state_input)
            # Rewards
            reward_shape = [None, len(self.actionSpaces[t])]
            reward_input = tf.placeholder(dtype=tf.float32, shape=reward_shape, name="rewards_{0}".format(t))
            self.rewards.append(reward_input)
        # Build policy generating networks; self.policies are filled.
        self.build_policy_networks()

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

    def sample_trajectories(self, sess, data, state_sample_count, samples_per_state):
        pass
        # history = []
        # total_sample_count = data.labelList.shape[0]
        # # Sample from s1 ~ p(s1)
        # sample_indices = np.random.choice(total_sample_count, state_sample_count, replace=False)
        # sample_indices = np.repeat(sample_indices, repeats=samples_per_state)
        # for t in range(self.trajectoryMaxLength):




            # # Sample actions from the policy pi_{t}(a_t|history(t))
            # feed_dict = self.prepare_sampling_feed_dict(curr_time_step=t)




        # for t in range(self.trajectoryMaxLength):
        #     # Sample actions from the policy pi_{t}(a_t|history(t))
        #     feed_dict = self.prepare_sampling_feed_dict(curr_time_step=t)
        #




            # sess.run(self.policies[t], feed_dict)



