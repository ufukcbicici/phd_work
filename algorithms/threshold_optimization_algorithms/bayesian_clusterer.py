import tensorflow as tf
import numpy as np

from algorithms.threshold_optimization_algorithms.deep_q_networks.dqn_networks import DeepQNetworks
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite


class BayesianClusterer:
    def __init__(self, network, routing_data, cluster_count, fc_layers, feature_name="pre_branch_feature"):
        self.network = network
        self.rootNode = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        self.routingData = routing_data
        self.featureName = feature_name
        with tf.variable_scope("clusterer"):
            assert len(self.rootNode) == 1
            self.rootNode = self.rootNode[0]
            feature_arr = self.routingData.get_dict(self.featureName)[self.rootNode.index]
            entry_shape = list(feature_arr.shape)
            entry_shape[0] = None
            self.netInput = tf.placeholder(dtype=tf.float32, shape=entry_shape, name="net_input")
            self.flattenedInput = DeepQNetworks.global_average_pooling(net_input=self.netInput)
            # net_shape = net.get_shape().as_list()
            # self.flattenedInput = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1],
            #                                      strides=[1, 1, 1, 1], padding='VALID')
            # self.flattenedInput = tf.squeeze(self.flattenedInput)
            net = self.flattenedInput
            hidden_layers = [fc_dim for fc_dim in fc_layers]
            hidden_layers.append(cluster_count)
            for layer_id, layer_dim in enumerate(hidden_layers):
                curr_dim = net.get_shape().as_list()[-1]
                W, b = FashionCignLite.get_affine_layer_params(
                    layer_shape=[curr_dim, layer_dim],
                    W_name="{0}_W_{1}".format("clusterer", layer_id),
                    b_name="{0}_b_{1}".format("clusterer", layer_id))
                net = FastTreeNetwork.fc_layer(x=net, W=W, b=b, node=self.rootNode)
                if layer_id < len(hidden_layers) - 1:
                    net = tf.nn.relu(net)
                else:
                    self.clustererOutput = tf.nn.softmax(net)

    def get_cluster_scores(self, sess, indices):
        feature_arr = self.routingData.get_dict(self.featureName)[self.rootNode.index]
        X = feature_arr[indices]
        results = sess.run([self.clustererOutput], feed_dict={self.netInput: X})
        cluster_scores = results[0]
        return cluster_scores


