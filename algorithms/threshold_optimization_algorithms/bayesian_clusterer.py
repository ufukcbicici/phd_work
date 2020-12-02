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
        self.globalStep = None
        self.optimizer = None
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
            # Optimizer
            self.scoresMatrix = tf.placeholder(dtype=tf.float32, shape=[None, cluster_count], name="scores_matrix")
            self.weightedScores = self.clustererOutput * self.scoresMatrix
            self.weightedScoresVector = tf.reduce_sum(self.weightedScores, axis=1)
            self.totalScore = tf.reduce_mean(self.weightedScoresVector)

    def get_cluster_scores(self, sess, features, batch_size=10000):
        cluster_scores_list = []
        for batch_idx in range(0, features.shape[0], batch_size):
            X_batch = features[batch_idx: batch_idx + batch_size]
            cluster_scores_batch = sess.run([self.clustererOutput], feed_dict={self.netInput: X_batch})[0]
            cluster_scores_list.append(cluster_scores_batch)
        cluster_scores = np.concatenate(cluster_scores_list, axis=0)
        return cluster_scores

    def optimize_clustering(self, sess, features, scores, accuracies, batch_size=256, iteration_count=10000):
        # Detect classes which are detected by some clusters
        # cluster_count = accuracies.shape[1]
        # accuracy_sums = np.sum(accuracies, axis=1)
        # arr_A = accuracy_sums > 0
        # arr_B = accuracy_sums < cluster_count
        # cluster_target_filter = np.logical_and(arr_A, arr_B)
        # features = features[cluster_target_filter]
        # scores = scores[cluster_target_filter]
        # X = feature_arr[indices]
        assert features.shape[0] == scores.shape[0]
        losses = []
        # Set up a new solver
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer().minimize(-self.totalScore, global_step=self.globalStep)
        trainable_variables = set(tf.trainable_variables())
        non_trainable_variables = [vr for vr in tf.global_variables() if vr not in trainable_variables]
        sess.run(tf.variables_initializer(non_trainable_variables))
        for iteration_id in range(iteration_count):
            sample_indices = np.random.choice(np.arange(features.shape[0]), size=batch_size, replace=True)
            X = features[sample_indices]
            scores_sample = scores[sample_indices]
            results = sess.run([self.totalScore, self.optimizer],
                               feed_dict={self.netInput: X, self.scoresMatrix: scores_sample})
            losses.append(results[0])
            if len(losses) == 10:
                curr_score = np.mean(np.array(losses))
                losses = []
                print("Bayesian Clusterer Iteration Id:{0} Curr Score:{1}".format(iteration_id, curr_score))










