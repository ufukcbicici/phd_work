import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from algorithms.accuracy_calculator import AccuracyCalculator
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_node import JungleNode
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.global_params import Optimizer


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        assert len(node_build_funcs) == len(h_funcs) + 1
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        curr_index = 0
        self.batchSize = tf.placeholder(name="batch_size", dtype=tf.int64)
        self.depthToNodesDict = {}
        self.hFuncs = h_funcs
        self.currentGraph = tf.get_default_graph()
        self.batchIndices = tf.cast(tf.range(self.batchSize), dtype=tf.int32)
        # self.decisionNoiseFactor = tf.placeholder(name="decision_noise_factor", dtype=tf.float32)
        # Create Trellis structure. Add a h node to every non-root and non-leaf layer.
        degree_list = [degree if depth == len(degree_list) - 1 else degree + 1 for depth, degree in
                       enumerate(degree_list)]
        assert degree_list[0] == 2
        assert degree_list[-1] == 1
        for depth, num_of_nodes in enumerate(degree_list):
            # root node, F_nodes, leaf nodes and H_node
            for index_in_depth in range(num_of_nodes):
                if depth < len(degree_list) - 1 and index_in_depth == num_of_nodes - 1:
                    node_type = NodeType.h_node
                elif depth == 0 and index_in_depth == 0:
                    assert num_of_nodes == 2
                    node_type = NodeType.root_node
                elif depth == len(degree_list) - 1:
                    node_type = NodeType.leaf_node
                else:
                    node_type = NodeType.f_node
                curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
                self.nodes[curr_index] = curr_node
                curr_index += 1
                if depth not in self.depthToNodesDict:
                    self.depthToNodesDict[depth] = []
                self.depthToNodesDict[depth].append(curr_node)
        # Build network as a DAG
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        self.build_network()
        self.build_optimizer()
        self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}
        # self.print_trellis_structure()

    def get_session(self):
        sess = tf.Session(graph=self.currentGraph)
        return sess

    def build_network(self):
        # Each H node will have the F nodes and the root node in the same layer and the H node in the previous layer
        # as the parents.
        # Each F node and leaf node have the H node in the previous layer as the parent.
        self.dagObject = Dag()
        for node in self.nodes.values():
            print(node.nodeType)
            if node.nodeType == NodeType.root_node:
                continue
            elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
                parent_h_nodes = [candidate_node for candidate_node in self.nodes.values()
                                  if candidate_node.depth == node.depth - 1
                                  and candidate_node.nodeType == NodeType.h_node]
                assert len(parent_h_nodes) == 1
                parent_h_node = parent_h_nodes[0]
                self.dagObject.add_edge(parent=parent_h_node, child=node)
            else:
                assert node.nodeType == NodeType.h_node
                parent_nodes = [candidate_node for candidate_node in self.nodes.values()
                                if (candidate_node.depth == node.depth
                                    and (candidate_node.nodeType == NodeType.f_node or
                                         candidate_node.nodeType == NodeType.root_node)) or
                                (candidate_node.depth == node.depth - 1 and candidate_node.nodeType == NodeType.h_node)]
                for parent_node in parent_nodes:
                    self.dagObject.add_edge(parent=parent_node, child=node)
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        # Build auxillary variables
        self.thresholdFunc(network=self)
        # Build node computational graphs
        for node in self.topologicalSortedNodes:
            # if node.depth > 3 or (node.depth == 3 and node.nodeType == NodeType.h_node):
            print("Building node {0}.".format(node.index))
            # if node.depth > 1:
            #     continue
            if node.nodeType == NodeType.root_node or node.nodeType == NodeType.f_node or \
                    node.nodeType == NodeType.leaf_node:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
                assert node.F_output is not None and node.H_output is None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
                node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = self.labelTensor
            elif node.nodeType == NodeType.h_node:
                self.hFuncs[node.depth](node=node, network=self)
                # assert node.F_output is not None and node.H_output is not None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
                node.evalDict[UtilityFuncs.get_variable_name(name="H_output", node=node)] = node.H_output
        # Build the network eval dict
        self.evalDict = {}
        for node in self.topologicalSortedNodes:
            for k, v in node.evalDict.items():
                assert k not in self.evalDict
                self.evalDict[k] = v

    def build_optimizer(self):
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        # Build optimizer
        self.globalCounter = tf.Variable(0, trainable=False)
        boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
        values = [GlobalConstants.INITIAL_LR]
        values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
        self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
        # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
        with tf.control_dependencies(self.extra_update_ops):
            # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
            #                                                                              global_step=self.globalCounter)
            self.optimizer = self.get_solver()

    def get_solver(self):
        if GlobalConstants.OPTIMIZER_TYPE == Optimizer.Adam:
            return tf.train.AdamOptimizer().minimize(self.finalLoss, global_step=self.globalCounter)
        elif GlobalConstants.OPTIMIZER_TYPE == Optimizer.Momentum:
            return tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
                                                                               global_step=self.globalCounter)

    def build_decision_loss(self):
        decision_losses = []
        for node in self.topologicalSortedNodes:
            if node.nodeType == NodeType.h_node and node.infoGainLoss is not None:
                decision_losses.append(node.infoGainLoss)
        self.decisionLoss = self.decisionLossCoefficient * tf.add_n(decision_losses)

    def build_main_loss(self):
        primary_losses = []
        for node in self.topologicalSortedNodes:
            if node.nodeType != NodeType.leaf_node:
                continue
            primary_losses.extend(node.lossList)
        self.mainLoss = tf.add_n(primary_losses)

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
            sampled_indices = self.sample_from_categorical(probs=p_F_given_x, batch_size=self.batchSize,
                                                           category_count=tf.constant(node_degree))
            arg_max_indices = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
            indices_tensor = tf.where(self.isTrain > 0, sampled_indices, arg_max_indices)
            # indices_tensor = sampled_indices
            # Step 4: Apply partitioning for corresponding F nodes in the same layer.
            node.conditionIndices = tf.dynamic_partition(data=self.batchIndices,
                                                         partitions=indices_tensor,
                                                         num_partitions=node_degree)
            node.F_output = tf.dynamic_partition(data=node.F_input, partitions=indices_tensor,
                                                 num_partitions=node_degree)
            node.H_output = tf.dynamic_partition(data=node.H_output, partitions=indices_tensor,
                                                 num_partitions=node_degree)
            node.labelTensor = tf.dynamic_partition(data=self.labelTensor, partitions=indices_tensor,
                                                    num_partitions=node_degree)
            # Reporting
            node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
            node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
            node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
            node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
            node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
            node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
            node.evalDict[UtilityFuncs.get_variable_name(name="condition_indices", node=node)] = node.conditionIndices
            node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = node.labelTensor
            node.evalDict[UtilityFuncs.get_variable_name(name="indices_tensor", node=node)] = indices_tensor
        else:
            node.conditionIndices = [self.batchIndices]
            node.F_output = [node.F_input]
            node.H_output = [node.H_output]
            node.labelTensor = [self.labelTensor]
            node.evalDict[UtilityFuncs.get_variable_name(name="indices_tensor", node=node)] = \
                tf.zeros(shape=self.batchSize, dtype=tf.int32)

    def apply_loss_jungle(self, node, final_feature):
        assert len(final_feature.get_shape().as_list()) == 2
        final_feature_dim = final_feature.get_shape().as_list()[-1]
        fc_softmax_weights = tf.Variable(
            tf.truncated_normal([final_feature_dim, self.labelCount], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="fc_softmax_weights", node=node))
        fc_softmax_biases = tf.Variable(tf.constant(0.1, shape=[self.labelCount],
                                                    dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="fc_softmax_biases", node=node))
        final_feature, logits = self.apply_loss(node=node, final_feature=final_feature,
                                                softmax_weights=fc_softmax_weights,
                                                softmax_biases=fc_softmax_biases)
        node.evalDict[self.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        assert len(node.lossList) == 1
        node.F_output = logits

    def get_node_sibling_index(self, node):
        sibling_nodes = [node for node in self.depthToNodesDict[node.depth]
                         if node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node or
                         node.nodeType == NodeType.root_node]
        sibling_nodes = {node.index: order_index for order_index, node in
                         enumerate(sorted(sibling_nodes, key=lambda c_node: c_node.index))}
        sibling_order_index = sibling_nodes[node.index]
        return sibling_order_index

    def mask_input_nodes(self, node):
        if node.nodeType == NodeType.root_node:
            node.F_input = self.dataTensor
            node.H_input = None
            node.conditionIndices = self.batchIndices
            node.labelTensor = self.labelTensor
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            # For reporting
            node.sampleCountTensor = tf.size(node.labelTensor)
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
            node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = node.labelTensor
            node.evalDict[
                UtilityFuncs.get_variable_name(name="condition_indices", node=node)] = node.conditionIndices
        elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
            # raise NotImplementedError()
            parents = self.dagObject.parents(node=node)
            assert len(parents) == 1 and parents[0].nodeType == NodeType.h_node
            parent_node = parents[0]
            sibling_order_index = self.get_node_sibling_index(node=node)
            with tf.control_dependencies([
                parent_node.F_output[sibling_order_index], parent_node.H_output[sibling_order_index],
                parent_node.labelTensor[sibling_order_index], parent_node.conditionIndices[sibling_order_index]]):
                node.F_input = tf.identity(parent_node.F_output[sibling_order_index])
                node.H_input = tf.identity(parent_node.H_output[sibling_order_index])
                node.labelTensor = tf.identity(parent_node.labelTensor[sibling_order_index])
                # For reporting
                node.sampleCountTensor = tf.size(parent_node.conditionIndices[sibling_order_index])
                is_used = tf.cast(node.sampleCountTensor, tf.float32) > 0.0
                node.isOpenIndicatorTensor = tf.where(is_used, 1.0, 0.0)
                node.conditionIndices = tf.identity(parent_node.conditionIndices[sibling_order_index])
                node.evalDict[self.get_variable_name(name="sample_count", node=node)] = node.sampleCountTensor
                node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
                node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = node.labelTensor
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="condition_indices", node=node)] = node.conditionIndices

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

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
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
                     self.batchSize: GlobalConstants.CURR_BATCH_SIZE}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_PROB
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                feed_dict[self.decisionDropoutKeepProb] = GlobalConstants.DECISION_DROPOUT_PROB
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
        else:
            feed_dict[self.classificationDropoutKeepProb] = 1.0
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                feed_dict[self.decisionDropoutKeepProb] = 1.0
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=False)
        return feed_dict

    def update_params(self, sess, dataset, epoch, iteration):
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
        # Unit Test for Unified Batch Normalization
        # if GlobalConstants.USE_VERBOSE:
        #     self.verbose_update(eval_dict=results[-1])
        # Unit Test for Unified Batch Normalization
        return lr, sample_counts, is_open_indicators

    def label_distribution_analysis(self,
                                    run_id,
                                    iteration,
                                    kv_rows,
                                    leaf_true_labels_dict,
                                    dataset,
                                    dataset_type):
        label_count = dataset.get_label_count()
        label_distribution = np.zeros(shape=(label_count,))
        for node in self.topologicalSortedNodes:
            if not node.nodeType == NodeType.leaf_node:
                continue
            true_labels = leaf_true_labels_dict[node.index]
            for l in range(label_count):
                label_distribution[l] = np.sum(true_labels == l)
                kv_rows.append((run_id, iteration, "{0} Leaf:{1} True Label:{2}".
                                format(dataset_type, node.index, l), np.asscalar(label_distribution[l])))

    def calculate_accuracy(self, calculation_type, sess, dataset, dataset_type, run_id, iteration):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_predicted_labels_dict = {}
        leaf_true_labels_dict = {}
        final_features_dict = {}
        info_gain_dict = {}
        branch_probs_dict = {}
        chosen_indices_dict = {}
        while True:
            results, _ = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                batch_sample_count = 0.0
                for node in self.topologicalSortedNodes:
                    if node.nodeType == NodeType.h_node:
                        node_degree = self.degreeList[node.depth + 1]
                        if node_degree == 1:
                            continue
                        info_gain = results[self.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.get_variable_name(name="p(n|x)", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict, key=node.index, array=branch_prob)
                        if node.index not in info_gain_dict:
                            info_gain_dict[node.index] = []
                        info_gain_dict[node.index].append(np.asscalar(info_gain))
                        continue
                    elif node.nodeType == NodeType.leaf_node:
                        posterior_probs = results[self.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = results[UtilityFuncs.get_variable_name(name="labelTensor", node=node)]
                        final_features = results[self.get_variable_name(name="final_feature_final", node=node)]
                        predicted_labels = np.argmax(posterior_probs, axis=1)
                        batch_sample_count += predicted_labels.shape[0]
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_predicted_labels_dict, key=node.index,
                                                             array=predicted_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=node.index,
                                                             array=true_labels)
                        UtilityFuncs.concat_to_np_array_dict(dct=final_features_dict, key=node.index,
                                                             array=final_features)
                if batch_sample_count != GlobalConstants.EVAL_BATCH_SIZE:
                    raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if dataset.isNewEpoch:
                break
        print("****************Dataset:{0}****************".format(dataset_type))
        kv_rows = []
        # Measure Information Gain
        total_info_gain = 0.0
        for k, v in info_gain_dict.items():
            avg_info_gain = sum(v) / float(len(v))
            print("IG_{0}={1}".format(k, -avg_info_gain))
            total_info_gain -= avg_info_gain
            kv_rows.append((run_id, iteration, "Dataset:{0} IG:{1}".format(dataset_type, k), avg_info_gain))
        kv_rows.append((run_id, iteration, "Dataset:{0} Total IG".format(dataset_type), total_info_gain))
        # Measure Branching Probabilities
        AccuracyCalculator.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                                branch_probs=branch_probs_dict, kv_rows=kv_rows)
        # Measure The Histogram of Branching Probabilities
        self.calculate_branch_probability_histograms(branch_probs=branch_probs_dict)
        # Measure Label Distribution
        self.label_distribution_analysis(run_id=run_id, iteration=iteration, kv_rows=kv_rows,
                                         leaf_true_labels_dict=leaf_true_labels_dict,
                                         dataset=dataset, dataset_type=dataset_type)
        # # Measure Accuracy
        overall_count = 0.0
        overall_correct = 0.0
        confusion_matrix_db_rows = []
        for node in self.topologicalSortedNodes:
            if not node.nodeType == NodeType.leaf_node:
                continue
            predicted = leaf_predicted_labels_dict[node.index]
            true_labels = leaf_true_labels_dict[node.index]
            if predicted.shape != true_labels.shape:
                raise Exception("Predicted and true labels counts do not hold.")
            correct_count = np.sum(predicted == true_labels)
            # Get the incorrect predictions by preparing a confusion matrix for each leaf
            sparse_confusion_matrix = {}
            for i in range(true_labels.shape[0]):
                true_label = true_labels[i]
                predicted_label = predicted[i]
                label_pair = (np.asscalar(true_label), np.asscalar(predicted_label))
                if label_pair not in sparse_confusion_matrix:
                    sparse_confusion_matrix[label_pair] = 0
                sparse_confusion_matrix[label_pair] += 1
            for k, v in sparse_confusion_matrix.items():
                confusion_matrix_db_rows.append((run_id, dataset_type.value, node.index, iteration, k[0], k[1], v))
            # Overall accuracy
            total_count = true_labels.shape[0]
            overall_correct += correct_count
            overall_count += total_count
            if total_count > 0:
                print("Leaf {0}: Sample Count:{1} Accuracy:{2}".format(node.index, total_count,
                                                                       float(correct_count) / float(total_count)))
            else:
                print("Leaf {0} is empty.".format(node.index))
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(overall_count, overall_correct / overall_count))
        total_accuracy = overall_correct / overall_count
        # Calculate modes
        # self.network.modeTracker.calculate_modes(leaf_true_labels_dict=leaf_true_labels_dict,
        #                                          dataset=dataset, dataset_type=dataset_type, kv_rows=kv_rows,
        #                                          run_id=run_id, iteration=iteration)
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return overall_correct / overall_count, confusion_matrix_db_rows

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
