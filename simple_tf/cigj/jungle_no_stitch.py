import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from algorithms.accuracy_calculator import AccuracyCalculator
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import JungleNode
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


class JungleNoStitch(Jungle):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        node.H_output = branching_feature
        node_degree = self.degreeList[node.depth + 1]
        if node_degree > 1:
            # Step 1: Create Hyperplanes
            ig_feature_size = node.H_output.get_shape().as_list()[-1]
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
            if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain, tf.bool))
            # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
            activations = tf.matmul(node.H_output, hyperplane_weights) + hyperplane_biases
            node.activationsDict[node.index] = activations
            decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
            p_F_given_x = tf.nn.softmax(decayed_activation)
            p_c_given_x = self.oneHotLabelTensor
            node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                      balance_coefficient=self.informationGainBalancingCoefficient)
            # Step 3:
            # If training: Sample Z from p(F|x) using Gumbel-Max trick
            # If testing: Pick Z = argmax_F p(F|x)
            category_count = tf.constant(node_degree)
            sampled_indices = self.sample_from_categorical(probs=p_F_given_x, batch_size=self.batchSize,
                                                           category_count=category_count)
            arg_max_indices = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
            sampled_one_hot_matrix = tf.one_hot(sampled_indices, category_count)
            arg_max_one_hot_matrix = tf.one_hot(arg_max_indices, category_count)
            node.conditionProbabilities = tf.where(self.isTrain > 0, sampled_one_hot_matrix, arg_max_one_hot_matrix)
            # Reporting
            node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
            node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
            node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
            node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
            node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
            node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
            node.evalDict[
                UtilityFuncs.get_variable_name(name="sampled_one_hot_matrix", node=node)] = sampled_one_hot_matrix
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
            node.evalDict[
                UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = node.conditionProbabilities
        else:
            node.conditionIndices = [self.batchIndices]
            node.F_output = [node.F_input]
            node.H_output = [node.H_output]
            node.labelTensor = [self.labelTensor]
            node.evalDict[UtilityFuncs.get_variable_name(name="indices_tensor", node=node)] = \
                tf.zeros(shape=self.batchSize, dtype=tf.int32)

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        # Layer 0 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 0
            node.F_input = parents[0].F_output
            node.H_input = None
            node.stitchedIndices = self.batchIndices
            node.stitchedLabels = self.labelTensor
        # Need stitching
        else:
            parent_node_degree = self.degreeList[node.depth]
            # Get all F nodes in the same layer
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=node)
                              if f_node.nodeType == NodeType.f_node]
            parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=node)
                              if h_node.nodeType == NodeType.h_node]
            assert len(parent_h_nodes) == 1
            parent_h_node = parent_h_nodes[0]
            parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
            assert all([f_node.H_output is None for f_node in parent_f_nodes])
            f_inputs = [node.F_output for node in parent_f_nodes]
            f_index_inputs = [node.conditionIndices for node in parent_f_nodes]
            f_label_inputs = [node.labelTensor for node in parent_f_nodes]
            control_dependencies = []
            control_dependencies.extend(parent_h_node.H_output)
            control_dependencies.extend(f_inputs)
            control_dependencies.extend(f_index_inputs)
            control_dependencies.extend(f_label_inputs)
            dbg_list = []
            with tf.control_dependencies(control_dependencies):
                # input_shapes = [tf.shape(x) for x in f_inputs]
                # shape_prods = [tf.reduce_prod(x) for x in input_shapes]
                # shape_sum = tf.add_n(shape_prods)
                # dbg_list.extend(input_shapes)
                # dbg_list.extend([tf.shape(x) for x in parent_h_node.conditionIndices])
                # dbg_list.extend([tf.shape(x) for x in parent_h_node.F_output])
                # dbg_list.extend([tf.shape(x) for x in parent_h_node.H_output])
                # dbg_list.extend([tf.shape(x) for x in parent_h_node.labelTensor])
                # assert_op = tf.Assert(tf.greater(shape_sum, 0), dbg_list, 1000)
                # with tf.control_dependencies([assert_op]):

                # node.F_input = tf.identity(parent_node.F_output[sibling_order_index])
                # node.H_input = tf.identity(parent_node.H_output[sibling_order_index])
                # node.labelTensor = tf.identity(parent_node.labelTensor[sibling_order_index])
                # node.conditionIndices = tf.identity(parent_node.conditionIndices[sibling_order_index])
                if parent_node_degree > 1:
                    f_shapes = []
                    indices_shapes = []
                    for f_input, condition_indices in zip(f_inputs, parent_h_node.conditionIndices):
                        f_shape = tf.shape(f_input)
                        indices_shape = tf.shape(condition_indices)
                        f_shapes.append(f_shape)
                        indices_shapes.append(indices_shape)
                        # f_shape = tf.shape(f_inputs)
                        # indices_shape = tf.shape(parent_h_node.conditionIndices)
                        # f_shape_print = tf.print(f_shape)
                        # indices_shape_print = tf.print(indices_shape)
                    # assert_op = tf.assert_equal(x=f_shape[0], y=indices_shape[0], data=[f_shape, indices_shape])
                    with tf.control_dependencies([tf.print("f_shapes:", *f_shapes),
                                                  tf.print("indices_shapes:", *indices_shapes)]):
                        node.F_input = tf.dynamic_stitch(indices=parent_h_node.conditionIndices, data=f_inputs)
                        node.H_input = tf.dynamic_stitch(indices=parent_h_node.conditionIndices,
                                                         data=parent_h_node.H_output)
                        node.stitchedIndices = tf.dynamic_stitch(indices=parent_h_node.conditionIndices,
                                                                 data=f_index_inputs)
                        node.stitchedLabels = tf.dynamic_stitch(indices=parent_h_node.conditionIndices,
                                                                data=f_label_inputs)
                else:
                    assert len(f_inputs) == 1
                    node.F_input = f_inputs[0]
                    assert len(parent_h_node.H_output) == 1
                    node.H_input = parent_h_node.H_output[0]
                    assert len(f_index_inputs) == 1
                    node.stitchedIndices = f_index_inputs[0]
                    assert len(f_label_inputs) == 1
                    node.stitchedLabels = f_label_inputs[0]
        node.evalDict[UtilityFuncs.get_variable_name(name="stitchedIndices", node=node)] = node.stitchedIndices
        node.evalDict[UtilityFuncs.get_variable_name(name="stitchedLabels", node=node)] = node.stitchedLabels
