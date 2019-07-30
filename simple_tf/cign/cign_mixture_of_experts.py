import tensorflow as tf
import numpy as np

from algorithms.info_gain import InfoGainLoss
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class CignMixtureOfExperts(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset)
        GlobalConstants.USE_UNIFIED_BATCH_NORM = False

    def get_parent_route_probs(self, node):
        parent_nodes = self.dagObject.parents(node=node)
        if len(parent_nodes) == 0:
            raise Exception("This is the root node.")
        assert len(parent_nodes) == 1
        parent_node = parent_nodes[0]
        p_n_given_x = parent_node.evalDict[self.get_variable_name(name="p(n|x)", node=parent_node)]
        child_nodes = self.dagObject.children(node=parent_node)
        child_nodes_sorted = sorted(child_nodes, key=lambda c_node: c_node.index)
        route_probs = None
        for child_index, child_node in enumerate(child_nodes_sorted):
            if child_node.index != node.index:
                continue
            route_probs = tf.expand_dims(p_n_given_x[:, child_index], axis=1)
            break
        assert route_probs is not None
        return route_probs

    def apply_decision(self, node, branching_feature):
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = tf.layers.batch_normalization(inputs=branching_feature,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain,
                                                                               tf.bool))
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
        activations = tf.matmul(branching_feature, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = node.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        category_count = tf.constant(node_degree)
        arg_max_indices = tf.argmax(p_n_given_x, axis=1, output_type=tf.int32)
        arg_max_one_hot_matrix = tf.one_hot(arg_max_indices, category_count)
        routing_probs = tf.where(self.isTrain > 0, p_n_given_x, arg_max_one_hot_matrix)
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        node.evalDict[self.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
        node.evalDict[self.get_variable_name(name="routing_probs", node=node)] = routing_probs

    # No actual masking for MoE models.
    def mask_input_nodes(self, node):
        print("Masking Node:{0}".format(node.index))
        if node.isRoot:
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            return None, None
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            route_probs = self.get_parent_route_probs(node=node)
            sample_count_tensor = tf.reduce_sum(route_probs)
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            parent_F = parent_node.fOpsList[-1]
            parent_H = parent_node.hOpsList[-1]
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = v
            node.labelTensor = parent_node.labelTensor
            node.indicesTensor = parent_node.indicesTensor
            node.oneHotLabelTensor = parent_node.oneHotLabelTensor
            return parent_F, parent_H

    def get_node_routing_probabilities(self, node):
        assert node.isLeaf
        curr_node = node
        routing_probs = []
        while True:
            parent_nodes = self.dagObject.parents(node=curr_node)
            if len(parent_nodes) == 0:
                break
            assert len(parent_nodes) == 1
            parent_node = parent_nodes[0]
            route_probs = self.get_parent_route_probs(node=curr_node)
            routing_probs.append(route_probs)
            curr_node = parent_node
        routing_matrix = tf.concat(values=routing_probs, axis=1)
        final_route_probs = tf.reduce_prod(routing_matrix, axis=1, keepdims=True)
        return final_route_probs

    def update_params(self, sess, dataset, epoch, iteration):
        update_results = super().update_params(sess=sess, dataset=dataset, epoch=epoch, iteration=iteration)
        if GlobalConstants.USE_VERBOSE:
            self.moe_unit_tests(eval_dict=update_results.evalDict)
        return update_results

    def moe_unit_tests(self, eval_dict):
        # Check that the routing probabilities are correct probability distributions
        routing_probs_list = []
        for node in self.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            routing_probs = eval_dict[self.get_variable_name(name="routing_probs", node=node)]
            routing_probs_list.append(routing_probs)
        routing_prob_matrix = np.concatenate(routing_probs_list, axis=1)
        sum_prob_matrix = np.sum(routing_prob_matrix, axis=1)
        assert np.allclose(sum_prob_matrix, np.ones_like(sum_prob_matrix))
