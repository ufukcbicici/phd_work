import tensorflow as tf
import numpy as np



class RoutingAccumulator:
    def __init__(self, network, routing_data, routingDecisions, leaf_index_dict, feature_dim=20,
                 use_posteriors_as_feature=True):
        self.network = network
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.routingData = routing_data
        self.routingDecisions = routingDecisions
        self.leafIndexDict = leaf_index_dict
        self.featureDim = feature_dim
        self.usePosteriorsAsFeature = use_posteriors_as_feature
        # Model objects
        self.routingMatrix = None
        self.activations = None
        self.posteriors = None

    def build_model(self):
        activations_dim = sum([self.network.dagObject.children(node=node) for node in self.leafNodes])
        posteriors_dim = self.routingData.posteriorProbs[self.leafNodes[0].index].shape[1] * len(self.leafNodes)
        self.routingMatrix = tf.placeholder(dtype=tf.int32, shape=[None, len(self.leafNodes)], name='routingMatrix')
        self.activations = tf.placeholder(dtype=tf.float32, shape=[None, activations_dim], name='activations')
        conditional_inputs = [(self.activations, True)]
        self.posteriors = tf.placeholder(dtype=tf.float32, shape=[None, posteriors_dim], name='posteriors')
        conditional_inputs.append((self.posteriors, self.usePosteriorsAsFeature))
        # Build the feature vector from activations and optionally with the posteriors
        feature_parts = []
        for tpl in conditional_inputs:
            if tpl[1] == False:
                continue
            feature_part = tf.layers.dense(inputs=tpl[0], units=self.featureDim, activation=tf.nn.relu)
            feature_parts.append(feature_part)
        feature_vector = tf.concat(feature_parts, axis=1)
        #








