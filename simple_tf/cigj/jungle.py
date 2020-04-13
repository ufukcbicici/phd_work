import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from algorithms.accuracy_calculator import AccuracyCalculator
from auxillary.constants import DatasetTypes
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_node import JungleNode
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants, TrainingUpdateResult
from algorithms.info_gain import InfoGainLoss
from simple_tf.global_params import Optimizer


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_funcs, grad_func, hyperparameter_func, residue_func, summary_func,
                 degree_list,
                 dataset, network_name):
        assert len(node_build_funcs) == len(h_funcs)
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func,
                         summary_func, degree_list, dataset, network_name)
        curr_index = 0
        self.batchSize = tf.placeholder(name="batch_size", dtype=tf.int64)
        self.depthToNodesDict = {}
        self.hFuncs = h_funcs
        self.currentGraph = tf.get_default_graph()
        self.batchIndices = tf.cast(tf.range(self.batchSize), dtype=tf.int32)
        # self.decisionNoiseFactor = tf.placeholder(name="decision_noise_factor", dtype=tf.float32)
        # Create Trellis structure. Add a h node to every non-root and non-leaf layer.
        degree_list = [degree + 1 for depth, degree in enumerate(degree_list)]
        assert degree_list[0] == 2
        for depth, num_of_nodes in enumerate(degree_list):
            # root node, F_nodes, leaf nodes and H_node
            for index_in_depth in range(num_of_nodes):
                if index_in_depth == num_of_nodes - 1:
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
        # self.print_trellis_structure()

    def get_session(self):
        sess = tf.Session(graph=self.currentGraph)
        return sess

    def build_network(self):
        # Each H node will have the F nodes and the root node in the same layer and the H node in the previous layer
        # as the parents.
        # Each F node and leaf node have the H node in the previous layer as the parent.
        self.dagObject = Dag()
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
                                         candidate_node.nodeType == NodeType.root_node or
                                         candidate_node.nodeType == NodeType.leaf_node)) or
                                (candidate_node.depth == node.depth - 1 and candidate_node.nodeType == NodeType.h_node)]
                for parent_node in parent_nodes:
                    self.dagObject.add_edge(parent=parent_node, child=node)
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        # Build auxillary variables
        # self.hyperparameterFunc(network=self)
        # Build node computational graphs
        for node in self.topologicalSortedNodes:
            # if node.depth > 3 or (node.depth == 3 and node.nodeType == NodeType.h_node):
            print("Building node {0}.".format(node.index))
            # if node.depth > 1:
            #     continue
            node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = self.labelTensor
            if node.nodeType == NodeType.root_node or node.nodeType == NodeType.f_node or \
                    node.nodeType == NodeType.leaf_node:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
                assert node.F_output is not None and node.H_output is None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
            elif node.nodeType == NodeType.h_node:
                self.hFuncs[node.depth](node=node, network=self)
                # assert node.F_output is not None and node.H_output is not None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
                node.evalDict[UtilityFuncs.get_variable_name(name="H_output", node=node)] = node.H_output
        # Build the network eval dict
        self.evalDict = {}
        self.build_optimizer()
        for node in self.topologicalSortedNodes:
            for k, v in node.evalDict.items():
                if v is None:
                    continue
                assert k not in self.evalDict
                self.evalDict[k] = v
        self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}

    def build_optimizer(self):
        # Build main classification loss
        self.build_main_loss()
        # Build information gain loss
        self.build_decision_loss()
        # Build regularization loss
        self.build_regularization_loss()
        # Final Loss
        self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        self.evalDict["mainLoss"] = self.mainLoss
        self.evalDict["regularizationLoss"] = self.regularizationLoss
        self.evalDict["decisionLoss"] = self.decisionLoss
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
            self.optimizer = self.get_solver()
            # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9)
            # self.decisionGradsOp = self.optimizer.compute_gradients(self.decisionLoss)
            # self.decisionGradsOp = [tpl for tpl in self.decisionGradsOp if tpl[0] is not None]
            # self.classificationGradsOp = self.optimizer.compute_gradients(self.mainLoss)
            # self.gradAndVarsOp = self.optimizer.compute_gradients(self.finalLoss)
            # self.trainOp = self.optimizer.apply_gradients(self.gradAndVarsOp, global_step=self.globalCounter)
            # activations_list = []
            # probs_list = []
            # z_probs_list = []
            # z_samples_list = []
            # for node in self.topologicalSortedNodes:
            #     if node.nodeType == NodeType.h_node:
            #         if UtilityFuncs.get_variable_name(name="activations", node=node) in node.evalDict:
            #             activation = node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)]
            #             activations_list.append(activation)
            #         if UtilityFuncs.get_variable_name(name="branch_probs", node=node) in node.evalDict:
            #             prob = node.evalDict[UtilityFuncs.get_variable_name(name="branch_probs", node=node)]
            #             probs_list.append(prob)
            #         if UtilityFuncs.get_variable_name(name="z_samples", node=node) in node.evalDict:
            #             z_samples = node.evalDict[UtilityFuncs.get_variable_name(name="z_samples", node=node)]
            #             z_samples_list.append(z_samples)
            #         if UtilityFuncs.get_variable_name(name="z_probs_matrix", node=node) in node.evalDict:
            #             z_probs_matrix = node.evalDict[UtilityFuncs.get_variable_name(name="z_probs_matrix", node=node)]
            #             z_probs_list.append(z_probs_matrix)
            # self.activationGrads = tf.gradients(ys=self.finalLoss, xs=activations_list)
            # self.activationGradsDecision = tf.gradients(ys=self.decisionLoss, xs=activations_list)
            # self.activationGradsClassification = tf.gradients(ys=self.mainLoss, xs=activations_list)
            # self.probGradsDecision = tf.gradients(ys=self.decisionLoss, xs=probs_list)
            # self.probGradsClassification = tf.gradients(ys=self.mainLoss, xs=probs_list)
            # self.zProbsGrads = tf.gradients(ys=self.finalLoss, xs=z_probs_list)
            # self.zSamplesGrads = tf.gradients(ys=self.finalLoss, xs=z_samples_list)

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
        leaf_h_node = sorted([node for node in self.topologicalSortedNodes
                              if node.nodeType == NodeType.h_node], key=lambda n: n.depth)[-1]
        softmax_output = leaf_h_node.F_output
        self.evalDict["softmax_output"] = softmax_output
        softmax_indices = tf.stack([self.batchIndices, tf.cast(self.labelTensor, tf.int32)], axis=1)
        self.evalDict["softmax_indices"] = softmax_indices
        selected_softmax_probs = tf.gather_nd(softmax_output, softmax_indices)
        self.evalDict["selected_softmax_probs"] = selected_softmax_probs
        zero_mask = 1e-30 * tf.cast(tf.equal(selected_softmax_probs, 0.0), tf.float32)
        self.evalDict["zero_mask"] = zero_mask
        stable_softmax_probs = selected_softmax_probs + zero_mask
        self.evalDict["stable_softmax_probs"] = stable_softmax_probs
        negative_log_likelihood = -tf.log(stable_softmax_probs)
        self.evalDict["negative_log_likelihood"] = negative_log_likelihood
        cross_entropy_loss = tf.reduce_mean(negative_log_likelihood)
        self.evalDict["cross_entropy_loss"] = cross_entropy_loss
        self.mainLoss = cross_entropy_loss

    def apply_loss_jungle(self, node, final_feature):
        assert len(final_feature.get_shape().as_list()) == 2
        final_feature_dim = final_feature.get_shape().as_list()[-1]
        fc_softmax_weights = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_softmax_weights", node=node),
            shape=[final_feature_dim, self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.truncated_normal([final_feature_dim, self.labelCount], stddev=0.1, seed=GlobalConstants.SEED,
                                            dtype=GlobalConstants.DATA_TYPE))
        fc_softmax_biases = UtilityFuncs.create_variable(
            name=UtilityFuncs.get_variable_name(name="fc_softmax_biases", node=node),
            shape=[self.labelCount],
            dtype=GlobalConstants.DATA_TYPE,
            initializer=tf.constant(0.1, shape=[self.labelCount],
                                    dtype=GlobalConstants.DATA_TYPE))
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = FastTreeNetwork.fc_layer(x=final_feature, W=fc_softmax_weights, b=fc_softmax_biases, node=node)
        node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
        node.evalDict[self.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
        node.F_output = node.evalDict[self.get_variable_name(name="posterior_probs", node=node)]

    def get_node_sibling_index(self, node):
        sibling_nodes = [node for node in self.depthToNodesDict[node.depth]
                         if node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node or
                         node.nodeType == NodeType.root_node]
        sibling_nodes = {node.index: order_index for order_index, node in
                         enumerate(sorted(sibling_nodes, key=lambda c_node: c_node.index))}
        sibling_order_index = sibling_nodes[node.index]
        return sibling_order_index

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        node.H_output = branching_feature
        if node.depth < len(self.degreeList) - 1:
            node_degree = self.degreeList[node.depth + 1]
            if node_degree > 1:
                # Step 1: Create Hyperplanes
                ig_feature_size = node.H_output.get_shape().as_list()[-1]
                hyperplane_weights = UtilityFuncs.create_variable(
                    name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
                    shape=[ig_feature_size, node_degree],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                                    dtype=GlobalConstants.DATA_TYPE))
                hyperplane_biases = UtilityFuncs.create_variable(
                    name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
                    shape=[node_degree],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
                if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                    node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                                  momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                                  training=tf.cast(self.isTrain, tf.bool))
                # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
                activations = FastTreeNetwork.fc_layer(x=node.H_output, W=hyperplane_weights, b=hyperplane_biases,
                                                       node=node)
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
                node.evalDict[UtilityFuncs.get_variable_name(name="branch_probs", node=node)] = p_F_given_x
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

    @staticmethod
    def multiply_tensor_with_branch_weights(weights, tensor):
        _w = tf.identity(weights)
        input_dim = len(tensor.get_shape().as_list())
        assert input_dim == 2 or input_dim == 4
        for _ in range(input_dim - 1):
            _w = tf.expand_dims(_w, axis=-1)
        weighted_tensor = tf.multiply(tensor, _w)
        return weighted_tensor

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        node.labelTensor = self.labelTensor
        # Layer 0 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 0
            node.F_input = parents[0].F_output
            node.H_input = None
        # Need stitching
        else:
            # Get all F nodes or Leaf nodes in the same layer
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=node)
                              if f_node.nodeType == NodeType.f_node or f_node.nodeType == NodeType.leaf_node]
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
                        Jungle.multiply_tensor_with_branch_weights(
                            weights=parent_h_node.conditionProbabilities[:, f_index],
                            tensor=f_input))
                node.F_input = tf.add_n(f_weighted_list)
                node.H_input = parent_h_node.H_output
        node.evalDict[UtilityFuncs.get_variable_name(name="F_input", node=node)] = node.F_input

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
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
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
        # grads_vars = sess.run([self.gradAndVarsOp], feed_dict=feed_dict)
        # for grads_vars in grads_vars:
        #     if np.any(np.isnan(grads_vars[0])):
        #         print("Gradient contains nan!")
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        # print("Before Update Iteration:{0}".format(iteration))
        results = sess.run(run_ops, feed_dict=feed_dict)

        # The following is for debug
        # # print("After Update Iteration:{0}".format(iteration))
        # self.gradAndVarsDict = results[0]
        # self.decisionGradsDict = results[-3]
        # self.classificationGradsDict = results[-2]
        # prob_grads_decision = results[-8]
        # prob_grads_classification = results[-7]
        # activation_grads = results[-6]
        # activation_grads_decision = results[-5]
        # activation_grads_classification = results[-4]
        # zero_count = np.sum(results[-1]["Node1_branch_probs"] == 0.0)
        # if zero_count > 0:
        #     print("We have zero!!!")
        # for grads in activation_grads_decision:
        #     if np.any(np.isnan(grads)):
        #         print("Gradient contains nan!")
        # for grads in activation_grads_classification:
        #     if np.any(np.isnan(grads)):
        #         print("Gradient contains nan!")
        # # for i in range(len(self.decisionGradsDict)):
        # #     a = self.decisionGradsDict[i][0]
        # #     b = self.classificationGradsDict[i][0]
        # #     c = self.gradAndVarsDict[i][0]
        # #     assert np.allclose(c, a + b)
        #
        # for grads_vars in self.gradAndVarsDict:
        #     if np.any(np.isnan(grads_vars[0])):
        #         print("Gradient contains nan!")
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        if GlobalConstants.USE_VERBOSE:
            update_results = TrainingUpdateResult(lr=lr, sample_counts=sample_counts,
                                                  is_open_indicators=is_open_indicators, eval_dict=results[-1])
        else:
            # Unit Test for Unified Batch Normalization
            update_results = TrainingUpdateResult(lr=lr, sample_counts=sample_counts,
                                                  is_open_indicators=is_open_indicators)
        return update_results

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

    @staticmethod
    def measure_h_node_label_distribution(arg_max_dict, labels_arr,
                                          dataset, dataset_type, kv_rows, run_id, iteration):
        label_count = dataset.get_label_count()
        for node_index, arg_max_indices in arg_max_dict.items():
            print("**************************After Node {0} distributions**************************".format(node_index))
            decisions = sorted(list(set(arg_max_indices)))
            for child_index in decisions:
                child_labels = labels_arr[arg_max_indices == child_index]
                label_distribution = np.zeros(shape=(label_count,))
                for l in range(label_count):
                    label_distribution[l] = np.sum(child_labels == l)
                    kv_rows.append(
                        (run_id, iteration, "{0} Leaf:{1} True Label:{2}".format(dataset_type, node_index, l),
                         np.asscalar(label_distribution[l])))
                label_distribution = label_distribution / float(len(child_labels))
                distribution_str = "Node {0} Child {1} ".format(node_index, child_index)
                for label_id, prob in enumerate(label_distribution):
                    distribution_str += "{0}:".format(label_id)
                    distribution_str += "%.4f" % prob
                    distribution_str += " "
                print("Node {0} weight:{1}".format(node_index, float(len(child_labels)) / float(len(labels_arr))))
                print(distribution_str)
                kv_rows.append(
                    (run_id, iteration, "Node {0} Child {1} Label Distribution".format(node_index, child_index),
                     distribution_str))
            print("X")

    def calculate_accuracy(self, sess, dataset, dataset_type, run_id, iteration,
                           posterior_entry_name="posterior_probs"):
        dataset.set_current_data_set_type(dataset_type=dataset_type, batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        leaf_posteriors_dict = {}
        true_labels_dict = {}
        final_features_dict = {}
        info_gain_dict = {}
        branch_probs_dict = {}
        arg_max_indices_dict = {}
        routing_probabilities_dict = {}
        final_posteriors_dict = {}
        # Data collection loop
        while True:
            results, _ = self.eval_network(sess=sess, dataset=dataset, use_masking=True)
            if results is not None:
                final_softmax_output = results["softmax_output"]
                UtilityFuncs.concat_to_np_array_dict(dct=final_posteriors_dict, key=0, array=final_softmax_output)
                for node in self.topologicalSortedNodes:
                    true_labels = results[UtilityFuncs.get_variable_name(name="labelTensor", node=node)]
                    UtilityFuncs.concat_to_np_array_dict(dct=true_labels_dict, key=node.index, array=true_labels)
                    if node.nodeType == NodeType.h_node:
                        if node.depth + 1 == len(self.degreeList):
                            continue
                        if self.degreeList[node.depth + 1] == 1:
                            continue
                        info_gain = results[self.get_variable_name(name="info_gain", node=node)]
                        branch_prob = results[self.get_variable_name(name="branch_probs", node=node)]
                        arg_max_indices = results[UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)]
                        routing_probabilities = \
                            results[UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)]
                        assert np.allclose(np.sum(routing_probabilities), GlobalConstants.EVAL_BATCH_SIZE)
                        UtilityFuncs.concat_to_np_array_dict(dct=branch_probs_dict, key=node.index, array=branch_prob)
                        UtilityFuncs.concat_to_np_array_dict(dct=arg_max_indices_dict, key=node.index,
                                                             array=arg_max_indices)
                        UtilityFuncs.concat_to_np_array_dict(dct=routing_probabilities_dict, key=node.index,
                                                             array=routing_probabilities)
                        if node.index not in info_gain_dict:
                            info_gain_dict[node.index] = []
                        info_gain_dict[node.index].append(np.asscalar(info_gain))
                        continue
                    elif node.nodeType == NodeType.leaf_node:
                        posterior_probs = results[self.get_variable_name(name="posterior_probs", node=node)]
                        final_features = results[self.get_variable_name(name="final_feature_final", node=node)]
                        UtilityFuncs.concat_to_np_array_dict(dct=final_features_dict, key=node.index,
                                                             array=final_features)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_posteriors_dict, key=node.index,
                                                             array=posterior_probs)
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
        # Measure h node label distribution
        # assert len(leaf_true_labels_dict) == 1
        label_tensors_list = list(true_labels_dict.values())
        assert all([np.array_equal(label_tensors_list[i], label_tensors_list[i + 1])
                    for i in range(len(label_tensors_list)-1)])
        true_labels = label_tensors_list[0]
        Jungle.measure_h_node_label_distribution(arg_max_dict=arg_max_indices_dict, labels_arr=true_labels,
                                                 dataset=dataset, dataset_type=dataset_type, run_id=run_id,
                                                 iteration=iteration, kv_rows=kv_rows)
        # Measure Branching Probabilities
        AccuracyCalculator.measure_branch_probs(run_id=run_id, iteration=iteration, dataset_type=dataset_type,
                                                branch_probs=branch_probs_dict, kv_rows=kv_rows)
        # Measure The Histogram of Branching Probabilities
        self.calculate_branch_probability_histograms(branch_probs=branch_probs_dict)
        # Measure Accuracy
        assert len(final_posteriors_dict.values()) == 1
        posterior_matrix = list(final_posteriors_dict.values())[0]
        predicted_labels = np.argmax(posterior_matrix, axis=1)
        accuracy_vector = (predicted_labels == true_labels).astype(np.float32)
        accuracy = np.mean(accuracy_vector)
        # Prepare the confusion matrix
        cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(accuracy_vector.shape[0], accuracy))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
        return accuracy, cm

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        is_evaluation_epoch_at_report_period = \
            epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
            and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
        is_evaluation_epoch_before_ending = \
            epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
        if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
            training_accuracy, training_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                        run_id=run_id,
                                        iteration=iteration)
            validation_accuracy, validation_confusion = \
                self.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                        run_id=run_id,
                                        iteration=iteration)
            DbLogger.write_into_table(
                rows=[(run_id, iteration, epoch_id, np.asscalar(training_accuracy),
                       np.asscalar(validation_accuracy), 0.0,
                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)

    def get_explanation_string(self):
        explanation = ""
        explanation += "TOTAL_EPOCH_COUNT:{0}\n".format(GlobalConstants.TOTAL_EPOCH_COUNT)
        explanation += "EPOCH_COUNT:{0}\n".format(GlobalConstants.EPOCH_COUNT)
        explanation += "EPOCH_REPORT_PERIOD:{0}\n".format(GlobalConstants.EPOCH_REPORT_PERIOD)
        explanation += "BATCH_SIZE:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "EVAL_BATCH_SIZE:{0}\n".format(GlobalConstants.EVAL_BATCH_SIZE)
        explanation += "USE_MULTI_GPU:{0}\n".format(GlobalConstants.USE_MULTI_GPU)
        explanation += "USE_SAMPLING_CIGN:{0}\n".format(GlobalConstants.USE_SAMPLING_CIGN)
        explanation += "USE_RANDOM_SAMPLING:{0}\n".format(GlobalConstants.USE_RANDOM_SAMPLING)
        explanation += "USE_SCALED_GRADIENTS:{0}\n".format(GlobalConstants.USE_SCALED_GRADIENTS)
        explanation += "LR SCHEDULE:{0}\n".format(GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation())
        single_path_cost = 0.0
        for depth, node_list in self.depthToNodesDict.items():
            non_h_nodes_cost = np.mean(np.array([node.macCost for node in node_list
                                                 if node.nodeType != NodeType.h_node]))
            h_node = [node for node in node_list if node.nodeType == NodeType.h_node]
            assert len(h_node) == 1
            h_node_cost = h_node[0].macCost
            layer_cost = non_h_nodes_cost + h_node_cost
            explanation += "Layer {0} Mac Cost:{1}\n".format(depth, layer_cost)
            single_path_cost += layer_cost
        explanation += "Mac Cost:{0}\n".format(single_path_cost)
        explanation += "Mac Cost per Nodes:{0}\n".format(self.nodeCosts)
        explanation += "Optimizer:{0}".format(self.optimizer)
        return explanation

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