import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class CignResnet50:
    def __init__(self):
        pass

    def apply_router_transformation(self, net, node, decision_feature_size):
        h_net = net
        net_shape = h_net.get_shape().as_list()
        # Global Average Pooling
        h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        net_shape = h_net.get_shape().as_list()
        h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        feature_size = h_net.get_shape().as_list()[-1]
        # MultiGPU OK
        fc_h_weights = UtilityFuncs.create_variable(
            name=self.get_variable_name(name="fc_decision_weights", node=node),
            shape=[feature_size, decision_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal(
            [feature_size, decision_feature_size], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
        # MultiGPU OK
        fc_h_bias = UtilityFuncs.create_variable(
            name=self.get_variable_name(name="fc_decision_bias", node=node),
            shape=[decision_feature_size],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE))
        h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
        h_net = tf.nn.relu(h_net)
        h_net = tf.nn.dropout(h_net, keep_prob=self.decisionDropoutKeepProb)
        ig_feature = h_net
        node.hOpsList.extend([ig_feature])
        # Decisions
        if GlobalConstants.USE_UNIFIED_BATCH_NORM:
            self.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
        else:
            self.apply_decision(node=node, branching_feature=ig_feature)