import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


class JungleNoStitch(Jungle):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)
        # self.unitTestList = [self.test_stitching]

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
                UtilityFuncs.get_variable_name(name="sampled_indices", node=node)] = sampled_indices
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)] = arg_max_indices
            node.evalDict[
                UtilityFuncs.get_variable_name(name="sampled_one_hot_matrix", node=node)] = sampled_one_hot_matrix
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
        else:
            node.conditionProbabilities = tf.ones_like(tensor=self.labelTensor, dtype=tf.float32)
        node.evalDict[
            UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = node.conditionProbabilities
        node.F_output = node.F_input

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        # Layer 0 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 0
            node.F_input = parents[0].F_output
            node.H_input = None
        # Need stitching
        else:
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
            # Get condition probabilities
            dependencies = []
            dependencies.extend(f_inputs)
            dependencies.append(parent_h_node.conditionProbabilities)
            with tf.control_dependencies(dependencies):
                f_weighted_list = []
                for f_index, f_input in enumerate(f_inputs):
                    f_weighted_list.append(
                        JungleNoStitch.multiply_tensor_with_branch_weights(
                            weights=parent_h_node.conditionProbabilities[:, f_index],
                            tensor=f_input))
                node.F_input = tf.add_n(f_weighted_list)
                node.H_input = parent_h_node.H_output
        node.evalDict[UtilityFuncs.get_variable_name(name="F_input", node=node)] = node.F_input

    @staticmethod
    def multiply_tensor_with_branch_weights(weights, tensor):
        _w = tf.identity(weights)
        input_dim = len(tensor.get_shape().as_list())
        assert input_dim == 2 or input_dim == 4
        for _ in range(input_dim - 1):
            _w = tf.expand_dims(_w, axis=-1)
        weighted_tensor = tf.multiply(tensor, _w)
        return weighted_tensor

    def mask_input_nodes(self, node):
        node.labelTensor = self.labelTensor
        if node.nodeType == NodeType.root_node:
            node.F_input = self.dataTensor
            node.H_input = None
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.conditionProbabilities = tf.ones_like(tensor=self.labelTensor, dtype=tf.float32)
            # For reporting
            node.sampleCountTensor = tf.reduce_sum(node.conditionProbabilities)
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            node.evalDict[UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = \
                node.conditionProbabilities
        elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
            # raise NotImplementedError()
            parents = self.dagObject.parents(node=node)
            assert len(parents) == 1 and parents[0].nodeType == NodeType.h_node
            parent_node = parents[0]
            sibling_order_index = self.get_node_sibling_index(node=node)
            with tf.control_dependencies([parent_node.F_output,
                                          parent_node.H_output,
                                          parent_node.conditionProbabilities]):
                node.F_input = tf.identity(parent_node.F_output)
                node.H_input = tf.identity(parent_node.H_output)
                if len(parent_node.conditionProbabilities.get_shape().as_list()) > 1:
                    node.conditionProbabilities = tf.identity(parent_node.conditionProbabilities[:,
                                                              sibling_order_index])
                else:
                    node.conditionProbabilities = tf.identity(parent_node.conditionProbabilities)
                # For reporting
                node.sampleCountTensor = tf.reduce_sum(node.conditionProbabilities)
                is_used = tf.cast(node.sampleCountTensor, tf.float32) > 0.0
                node.isOpenIndicatorTensor = tf.where(is_used, 1.0, 0.0)
                # node.conditionIndices = tf.identity(parent_node.conditionIndices[sibling_order_index])
                node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
                node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = \
                    node.conditionProbabilities

    def update_params_with_momentum(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.CURR_BATCH_SIZE)
        if minibatch is None:
            return None, None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                           is_train=True, use_masking=True)
        # Prepare result tensors to collect
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        # print("Before Update Iteration:{0}".format(iteration))
        results = sess.run(run_ops, feed_dict=feed_dict)
        # print("After Update Iteration:{0}".format(iteration))
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        # Unit Tests
        if GlobalConstants.USE_UNIT_TESTS:
            for test in self.unitTestList:
                test(results[-1])
        return lr, sample_counts, is_open_indicators

    # Unit test methods
    def test_stitching(self, eval_dict):
        h_nodes = [node for node in self.topologicalSortedNodes if node.nodeType == NodeType.h_node]
        for h_node in h_nodes:
            if len(self.dagObject.parents(node=h_node)) == 1:
                continue
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=h_node)
                              if f_node.nodeType == NodeType.f_node]
            parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
            parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=h_node)
                              if h_node.nodeType == NodeType.h_node]
            assert len(parent_h_nodes) == 1
            f_input = eval_dict[UtilityFuncs.get_variable_name(name="F_input", node=h_node)]
            condition_probabilities = eval_dict[UtilityFuncs.get_variable_name(name="conditionProbabilities",
                                                                               node=parent_h_nodes[0])]
            f_outputs_prev_layer = [eval_dict[UtilityFuncs.get_variable_name(name="F_output", node=node)]
                                    for node in parent_f_nodes]
            f_input_manual = np.zeros_like(f_input)
            for r in range(condition_probabilities.shape[0]):
                selected_node_idx = np.asscalar(np.argmax(condition_probabilities[r, :]))
                f_input_manual[r, :] = f_outputs_prev_layer[selected_node_idx][r, :]
            assert np.allclose(f_input, f_input_manual)




