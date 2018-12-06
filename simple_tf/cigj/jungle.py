from collections import deque

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cigj.jungle_node import JungleNode


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        curr_index = 0
        self.depthToNodesDict = {}
        self.hFuncs = h_funcs
        # Create Trellis structure. Add a h node to every non-root and non-leaf layer.
        degree_list = [degree if depth == 0 or depth == len(degree_list) - 1 else degree + 1 for depth, degree in
                       enumerate(degree_list)]
        assert degree_list[0] == 1
        assert degree_list[-1] == 1
        for depth, num_of_nodes in enumerate(degree_list):
            for i in range(num_of_nodes):
                if depth == 0:
                    node_type = NodeType.root_node
                elif depth == len(degree_list) - 1:
                    node_type = NodeType.leaf_node
                elif i == 0:
                    node_type = NodeType.h_node
                elif i > 0:
                    node_type = NodeType.f_node
                else:
                    raise Exception("Unknown node type.")
                curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
                self.nodes[curr_index] = curr_node
                curr_index += 1
                if depth not in self.depthToNodesDict:
                    self.depthToNodesDict[depth] = []
                self.depthToNodesDict[depth].append(curr_node)
        # Build network as a DAG
        self.build_network()
        # Build auxillary variables
        self.thresholdFunc(network=self)
        # Build the network eval dict
        for node in self.topologicalSortedNodes:
            for k, v in node.evalDict.items():
                assert k not in self.evalDict
                self.evalDict[k] = v

    def build_network(self):
        # Each H node will have the whole previous layer as the parent.
        # Each F node will have the H of the same layer as the parent.
        # Root has the Layer 1 H as its child.
        # Leaf will have the whole previous layer as the parent.
        for node in self.nodes.values():
            if node.nodeType == NodeType.root_node:
                continue
            elif node.nodeType == NodeType.h_node or node.nodeType == NodeType.leaf_node:
                assert node.depth - 1 in self.depthToNodesDict
                for parent_node in self.depthToNodesDict[node.depth - 1]:
                    self.dagObject.add_edge(parent=parent_node, child=node)
            elif node.nodeType == NodeType.f_node:
                h_nodes = [node for node in self.depthToNodesDict[node.depth] if node.nodeType == NodeType.h_node]
                assert len(h_nodes) == 1
                parent = h_nodes[0]
                self.dagObject.add_edge(parent=parent, child=node)
            else:
                raise Exception("Unknown node type.")
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        # Build node computational graphs
        for node in self.topologicalSortedNodes:
            if node.depth > 1:
                break
            if node.nodeType == NodeType.root_node:
                self.nodeBuildFuncs[0](node=node, network=self)
                assert node.F_output is not None and node.H_output is None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
            elif node.nodeType == NodeType.h_node:
                self.hFuncs[node.depth - 1](node=node, network=self)
                assert node.F_output is not None and node.H_output is not None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
                node.evalDict[UtilityFuncs.get_variable_name(name="H_output", node=node)] = node.H_output
            elif node.nodeType == NodeType.f_node:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
                assert node.F_output is not None and node.H_output is None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        # Layer 1 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 1
            node.F_output = parents[0].F_output
            node.labelTensor = parents[0].labelTensor
            node.indicesTensor = parents[0].indicesTensor
            node.oneHotLabelTensor = parents[0].oneHotLabelTensor
            return parents[0].F_output, None
        # Need stitching
        else:
            raise NotImplementedError()

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        # Step 1: Create Hyperplanes
        node_degree = self.degreeList[node.depth]
        ig_feature_size = branching_feature.get_shape().as_list()[-1]
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = tf.layers.batch_normalization(inputs=branching_feature,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain,
                                                                               tf.bool))
        # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
        activations = tf.matmul(branching_feature, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1, ))
        p_F_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = self.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
        # Step 3: Sample from p(F|x) when training, select argmax_F p(F|x) during inference
        dist = tf.distributions.Categorical(probs=p_F_given_x)
        samples = dist.sample()
        one_hot_samples = tf.one_hot(indices=samples, depth=node_degree, axis=-1, dtype=tf.int64)
        one_hot_argmax = tf.one_hot(indices=tf.argmax(p_F_given_x, axis=1), depth=node_degree, axis=-1, dtype=tf.int64)
        one_hot_indices = tf.where(self.isTrain > 0, one_hot_samples, one_hot_argmax)
        node.evalDict[UtilityFuncs.get_variable_name(name="samples", node=node)] = samples
        node.evalDict[UtilityFuncs.get_variable_name(name="one_hot_samples", node=node)] = one_hot_samples
        node.evalDict[UtilityFuncs.get_variable_name(name="one_hot_indices", node=node)] = one_hot_indices
        # Step 4: Apply masking to corresponding F nodes in the same layer.
        child_F_nodes = [node for node in self.depthToNodesDict[node.depth] if node.nodeType == NodeType.f_node]
        child_F_nodes = sorted(child_F_nodes, key=lambda c_node: c_node.index)
        for index in range(len(child_F_nodes)):
            child_node = child_F_nodes[index]
            child_index = child_node.index
            node.maskTensors[child_index] = one_hot_indices[:, index]
            node.evalDict[UtilityFuncs.get_variable_name(name="mask_vector_{0}_{1}".format(index, child_index),
                                                         node=node)] = node.maskTensors[child_index]

    def mask_input_nodes(self, node):
        if node.nodeType == NodeType.root_node:
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            # For reporting
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        elif node.nodeType == NodeType.f_node:
            parents = self.dagObject.parents(node=node)
            assert len(parents) == 1 and parents[0].nodeType == NodeType.h_node
            parent_node = parents[0]
            mask_tensor = parent_node.maskTensors[node.index]
            sample_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.float32))
            node.isOpenIndicatorTensor = tf.where(sample_count_tensor > 0.0, 1.0, 0.0)
            # Mask all inputs (F, H)
            parent_F = tf.boolean_mask(parent_node.F_output, mask_tensor)
            parent_H = tf.boolean_mask(parent_node.H_output, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
            node.indicesTensor = tf.boolean_mask(parent_node.indicesTensor, mask_tensor)
            node.oneHotLabelTensor = tf.boolean_mask(parent_node.oneHotLabelTensor, mask_tensor)
            # For reporting
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            node.evalDict[self.get_variable_name(name="parent_F", node=node)] = parent_F
            node.evalDict[self.get_variable_name(name="parent_H", node=node)] = parent_H
            node.evalDict[self.get_variable_name(name="labelTensor", node=node)] = node.labelTensor
            node.evalDict[self.get_variable_name(name="indicesTensor", node=node)] = node.indicesTensor
            node.evalDict[self.get_variable_name(name="oneHotLabelTensor", node=node)] = node.oneHotLabelTensor
            return parent_F, parent_H
        else:
            raise Exception("Unknown node type.")

    def get_softmax_decays(self, feed_dict, iteration, update):
        for node in self.topologicalSortedNodes:
            if node.nodeType != NodeType.h_node:
                continue
            # Decay for Softmax
            decay = node.softmaxDecayCalculator.value
            if update:
                feed_dict[node.softmaxDecay] = decay
                UtilityFuncs.print("{0} value={1}".format(node.softmaxDecayCalculator.name, decay))
                # Update the Softmax Decay
                node.softmaxDecayCalculator.update(iteration=iteration + 1)
            else:
                feed_dict[node.softmaxDecay] = GlobalConstants.SOFTMAX_TEST_TEMPERATURE

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking, batch_size):
        feed_dict = {self.dataTensor: minibatch.samples,
                     self.labelTensor: minibatch.labels,
                     self.indicesTensor: minibatch.indices,
                     self.oneHotLabelTensor: minibatch.one_hot_labels,
                     # self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     # self.isDecisionPhase: int(is_decision_phase),
                     self.isTrain: int(is_train),
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration,
                     self.batchSize: batch_size}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_PROB
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                # self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
                feed_dict[self.decisionDropoutKeepProb] = GlobalConstants.DECISION_DROPOUT_PROB
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
        else:
            feed_dict[self.classificationDropoutKeepProb] = 1.0
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                feed_dict[self.decisionDropoutKeepProb] = 1.0
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=False)
        return feed_dict

    # For debugging
    def print_trellis_structure(self):
        fig, ax = plt.subplots()
        # G = self.dagObject.dagObject
        node_radius = 0.05
        node_circles = []
        node_positions = {}
        # Draw Nodes as Vertices (Circles)
        for curr_depth in range(self.depth):
            nodes_of_curr_depth = self.depthToNodesDict[curr_depth]
            if len(nodes_of_curr_depth) > 1:
                horizontal_step_size = (1.0 - 2 * node_radius) / float(len(nodes_of_curr_depth) - 1.0)
                vertical_step_size = (1.0 - 2 * node_radius) / float(len(self.depthToNodesDict) - 1.0)
            else:
                horizontal_step_size = 0.0
                vertical_step_size = 0.0
            for index_in_depth, node in enumerate(nodes_of_curr_depth):
                if node.nodeType == NodeType.root_node:
                    node_circles.append(plt.Circle((0.5, 1.0 - node_radius), node_radius, color='r'))
                    node_positions[node] = (0.5, 1.0 - node_radius)
                elif node.nodeType == NodeType.leaf_node:
                    node_circles.append(plt.Circle((0.5, node_radius), node_radius, color='y'))
                    node_positions[node] = (0.5, node_radius)
                elif node.nodeType == NodeType.f_node:
                    node_circles.append(plt.Circle((node_radius + index_in_depth * horizontal_step_size,
                                                    1.0 - node_radius - curr_depth * vertical_step_size), node_radius,
                                                   color='b'))
                    node_positions[node] = (node_radius + index_in_depth * horizontal_step_size,
                                            1.0 - node_radius - curr_depth * vertical_step_size)
                elif node.nodeType == NodeType.h_node:
                    node_circles.append(plt.Circle((node_radius + index_in_depth * horizontal_step_size,
                                                    1.0 - node_radius - curr_depth * vertical_step_size), node_radius,
                                                   color='g'))
                    node_positions[node] = (node_radius + index_in_depth * horizontal_step_size,
                                            1.0 - node_radius - curr_depth * vertical_step_size)
                else:
                    raise Exception("Unknown node type.")
        for circle in node_circles:
            ax.add_artist(circle)
        # Draw Edges as Arrows
        for edge in self.dagObject.get_edges():
            source = edge[0]
            destination = edge[1]
            ax.arrow(node_positions[source][0], node_positions[source][1],
                     node_positions[destination][0] - node_positions[source][0],
                     node_positions[destination][1] - node_positions[source][1],
                     head_width=0.01, head_length=0.01, fc='k', ec='k',
                     length_includes_head=True)
        # Draw node texts
        for node in self.topologicalSortedNodes:
            node_pos = node_positions[node]
            ax.text(node_pos[0], node_pos[1], "{0}".format(node.index), fontsize=16, color="c")
        plt.show()
        UtilityFuncs.print("X")

    # def apply_decision(self):
    #     pass
