import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.policy_gradient_algorithms.policy_gradients_network import \
    PolicyGradientsNetwork


class TreeDepthPolicyNetwork(PolicyGradientsNetwork):
    def __init__(self, action_spaces, state_shapes, l2_lambda):
        super().__init__(action_spaces, state_shapes, l2_lambda)
        