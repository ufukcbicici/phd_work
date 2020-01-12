import numpy as np
import tensorflow as tf
from algorithms.threshold_optimization_algorithms.combinatorial_routing_optimizer import CombinatorialRoutingOptimizer
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork


class PolicyGradientsNetwork:
    def __init__(self, action_spaces, state_shapes, l2_lambda):
        self.actionSpaces = action_spaces
        self.l2Lambda = l2_lambda
        self.paramL2Norms = {}
        self.l2Loss = None
        self.trajectoryMaxLength = len(action_spaces)
        assert len(state_shapes) == self.trajectoryMaxLength
        self.stateShapes = state_shapes
        self.inputs = []
        self.policies = []

    def build_policy_networks(self):
        pass

    def build_policy_gradient_loss(self):
        pass

    def build_network(self):
        # State inputs
        for t in range(self.trajectoryMaxLength):
            input_shape = [None]
            input_shape.extend(self.stateShapes[t])
            state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
            self.inputs.append(state_input)
        # Build policy generating networks; self.policies are filled.
        self.build_policy_networks()



        # self.branchingStateVectors = branching_state_vectors
        # # self.hiddenLayers = [self.branchingStateVectors.shape[-1]]
        # # self.hiddenLayers.extend(hidden_layers)
        # self.hiddenLayers = hidden_layers
        # self.actionCount = int(action_space_size)
        # self.inputs = None
        # # Policy MLP
        # self.net = None
        # self.logits = None
        # self.pi = None
        # self.repeatedPolicy = None
        # self.logRepeatedPolicy = None
        # self.stateCount = None
        # self.categoryCount = None
        # # Policy Evaluation on the given data
        # self.rewards = None
        # self.weightedRewardMatrix = None
        # self.valueFunctions = None
        # self.policyValue = None
        # self.policySampleCount = None
        # self.policySamples = None
        # self.logSampledPolicies = None
        # self.rewardSamples = None
        # self.proxyValueVector = None
        # self.proxyPolicyValue = None
        # self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # self.learningRate = tf.constant(0.00001)
        # self.optimizer = None
        # self.l2Loss = None
        # self.totalLoss = None
        # self.l2Lambda = PolicyGradientsRoutingOptimizer.L2_LAMBDA
        # self.paramL2Norms = {}
        # # Build network
        # self.build_network()

    # def build_network(self):
    #     self.inputs = tf.placeholder(dtype=tf.float32,
    #                                  shape=[None, self.branchingStateVectors.shape[-1]], name="inputs")
    #     self.policySampleCount = tf.placeholder(dtype=tf.int32, name="policySampleCount")
    #     self.hiddenLayers.append(self.actionCount)
    #     # Policy MLP
    #     self.net = self.inputs
    #     for layer_id, layer_dim in enumerate(self.hiddenLayers):
    #         if layer_id < len(self.hiddenLayers) - 1:
    #             self.net = tf.layers.dense(inputs=self.net, units=layer_dim, activation=tf.nn.relu)
    #         else:
    #             self.net = tf.layers.dense(inputs=self.net, units=layer_dim, activation=None)
    #     self.logits = self.net
    #     self.pi = tf.nn.softmax(self.logits)
    #     # Policy Evaluation on the given data
    #     self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, self.actionCount], name="rewards")
    #     self.weightedRewardMatrix = self.pi * self.rewards
    #     self.valueFunctions = tf.reduce_sum(self.weightedRewardMatrix, axis=1)
    #     self.policyValue = tf.reduce_mean(self.valueFunctions)
    #     # Proxy Objective Function and maximizing it
    #     self.repeatedPolicy = tf.tile(self.pi, [self.policySampleCount, 1])
    #     self.logRepeatedPolicy = tf.log(self.repeatedPolicy)
    #     self.policySamples = FastTreeNetwork.sample_from_categorical_v2(probs=self.repeatedPolicy)
    #     prob_shape = tf.shape(self.logRepeatedPolicy)
    #     self.stateCount = tf.gather_nd(prob_shape, [0])
    #     self.categoryCount = tf.gather_nd(prob_shape, [1])
    #     self.policySamples = tf.stack([tf.range(0, self.stateCount, 1), self.policySamples], axis=1)
    #     self.logSampledPolicies = tf.gather_nd(self.logRepeatedPolicy, self.policySamples)
    #     self.rewardSamples = tf.gather_nd(self.rewards, self.policySamples)
    #     self.proxyValueVector = self.logSampledPolicies * self.rewardSamples
    #     self.proxyPolicyValue = tf.reduce_mean(self.proxyValueVector)
    #     self.get_l2_loss()
    #     self.totalLoss = (-1.0 * self.proxyPolicyValue) + self.l2Loss
    #     # self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)
    #     self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate). \
    #         minimize(self.totalLoss, global_step=self.globalStep)

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
            self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)
