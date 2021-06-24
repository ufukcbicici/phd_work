from collections import deque
from collections import Counter
import numpy as np
import tensorflow as tf
from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from simple_tf.uncategorized.node import Node
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.utilities import Utilities
import time


# tf.autograph.set_verbosity(10, True)


class Cign:
    def __init__(self,
                 input_dims,
                 class_count,
                 node_degrees,
                 decision_drop_probability,
                 classification_drop_probability,
                 decision_wd,
                 classification_wd,
                 information_gain_balance_coeff,
                 softmax_decay_controller,
                 bn_momentum=0.9):
        self.dagObject = Dag()
        self.nodes = {}
        self.degreeList = node_degrees
        self.networkDepth = len(self.degreeList)
        self.orderedNodesPerLevel = [[]] * (self.networkDepth + 1)
        self.topologicalSortedNodes = []
        self.innerNodes = []
        self.leafNodes = []
        self.isBaseline = None
        self.informationGainBalanceCoeff = information_gain_balance_coeff
        self.classCount = class_count
        self.decisionDropProbability = decision_drop_probability
        self.classificationDropProbability = classification_drop_probability
        self.decisionWd = decision_wd
        self.classificationWd = classification_wd
        self.bnMomentum = bn_momentum
        # Model-wise Tensorflow objects.
        self.inputs = tf.keras.Input(shape=input_dims, name="inputs")
        self.labels = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)
        self.batchSize = tf.shape(self.inputs)[0]
        self.batchIndices = tf.range(0, self.batchSize, 1)
        # Hyper-parameters
        # Node input-outputs
        # self.labelsDict = {}
        # self.batchIndicesDict = {}
        self.nodeOutputsDict = {}
        # Routing temperatures
        self.routingTemperatures = {}
        # Masked Batch Norm Layers
        self.maskedBatchNormOps = {}
        # Feed dict
        self.feedDict = {"inputs": self.inputs, "labels": self.labels}
        # # Information Gain Mask Matrices
        # self.igMaskMatricesDict = {}
        # # Secondary Routing Mask Matrices (Heuristic thresholding, Reinforcement Learning, Bayesian Optimization, etc.)
        # self.secondaryMaskMatricesDict = {}
        # Global evaluation dictionary
        self.evalDict = {}
        # Node builder functions
        self.nodeBuildFuncs = []
        # Softmax decay controllers
        self.softmaxDecayController = softmax_decay_controller
        # Classification losses
        self.classificationLossObjects = {}
        self.classificationLosses = {}
        # Routing Losses
        self.informationGainRoutingLosses = {}
        # Regularization Losses
        self.regularizationCoefficients = {}
        self.regularizationLoss = None
        # Model
        self.model = None
        self.model2 = None
        self.model3 = None

    def get_node_sibling_index(self, node):
        parent_nodes = self.dagObject.parents(node=node)
        if len(parent_nodes) == 0:
            return 0
        parent_node = parent_nodes[0]
        siblings_dict = {sibling_node.index: order_index for order_index, sibling_node in
                         enumerate(
                             sorted(self.dagObject.children(node=parent_node),
                                    key=lambda c_node: c_node.index))}
        sibling_index = siblings_dict[node.index]
        return sibling_index

    def build_tree(self):
        # Create itself
        curr_index = 0
        is_leaf = 0 == self.networkDepth
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        d = deque()
        d.append(root_node)
        # Create children if not leaf
        while len(d) > 0:
            # Dequeue
            curr_node = d.popleft()
            if not curr_node.isLeaf:
                for i in range(self.degreeList[curr_node.depth]):
                    new_depth = curr_node.depth + 1
                    is_leaf = new_depth == self.networkDepth
                    curr_index += 1
                    child_node = Node(index=curr_index, depth=new_depth, is_root=False, is_leaf=is_leaf)
                    self.nodes[curr_index] = child_node
                    self.dagObject.add_edge(parent=curr_node, child=child_node)
                    d.append(child_node)
        # Topological structures
        nodes_per_level_dict = {}
        for node in self.nodes.values():
            if node.depth not in nodes_per_level_dict:
                nodes_per_level_dict[node.depth] = []
            nodes_per_level_dict[node.depth].append(node)
            self.nodeOutputsDict[node.index] = {}
        for level in nodes_per_level_dict.keys():
            self.orderedNodesPerLevel[level] = sorted(nodes_per_level_dict[level], key=lambda n: n.index)
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        self.innerNodes = [node for node in self.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda nd: nd.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda nd: nd.index)

    def build_network(self):
        self.build_tree()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Build all operations in each node -> Level by level
        for level in range(len(self.orderedNodesPerLevel)):
            for node in self.orderedNodesPerLevel[level]:
                f_input, h_input, ig_mask = self.mask_inputs(node=node)
                self.nodeBuildFuncs[node.depth](self=self, node=node, f_input=f_input, h_input=h_input)
                # Build information gain based routing matrices after a level's nodes being built
                # (if not the final layer)
                if level < len(self.orderedNodesPerLevel) - 1:
                    # Inputs for the routing temperature and the masked batch normalization layer
                    self.routingTemperatures[node.index] = \
                        tf.keras.Input(shape=(), name="routingTemperature_{0}".format(node.index))
                    self.feedDict["routingTemperature_{0}".format(node.index)] = self.routingTemperatures[node.index]
                    self.maskedBatchNormOps[node.index] = MaskedBatchNormalization(momentum=self.bnMomentum)
                    self.apply_decision(node=node, ig_mask=ig_mask)
                else:
                    self.apply_classification_loss(node=node)
            # Build secondary routing matrices after a level's nodes being built (if not the final layer)
            if level < len(self.orderedNodesPerLevel) - 1:
                self.build_secondary_routing_matrices(level=level)
        # Register node outputs to the eval dict
        for node in self.topologicalSortedNodes:
            for output_name in self.nodeOutputsDict[node.index].keys():
                if self.nodeOutputsDict[node.index][output_name] is None:
                    continue
                self.evalDict[Utilities.get_variable_name(name="node_output_{0}".format(output_name), node=node)] = \
                    self.nodeOutputsDict[node.index][output_name]
        self.evalDict["batch_size"] = self.batchSize
        # Build the model
        self.build_final_model()

    def is_decision_variable(self, variable):
        if "scale" in variable.name or "shift" in variable.name or "hyperplane" in variable.name or \
                "gamma" in variable.name or "beta" in variable.name or "decision" in variable.name:
            return True
        else:
            return False

    def calculate_regularization_coefficients(self, trainable_variables):
        print("Num of trainable variables:{0}".format(len(trainable_variables)))
        l2_loss_list = []
        decayed_variables = []
        non_decayed_variables = []
        decision_variables = []
        for v in trainable_variables:
            is_decision_pipeline_variable = self.is_decision_variable(variable=v)
            # assert (not is_decision_pipeline_variable)
            loss_tensor = tf.nn.l2_loss(v)
            # self.evalDict["l2_loss_{0}".format(v.name)] = loss_tensor
            if "bias" in v.name or "shift" in v.name or "scale" in v.name or "gamma" in v.name or "beta" in v.name:
                # non_decayed_variables.append(v)
                # l2_loss_list.append(0.0 * loss_tensor)
                self.regularizationCoefficients[v.ref()] = 0.0
            else:
                # decayed_variables.append(v)
                if is_decision_pipeline_variable:
                    # decision_variables.append(v)
                    # l2_loss_list.append(self.decisionWd * loss_tensor)
                    self.regularizationCoefficients[v.ref()] = self.decisionWd
                else:
                    # l2_loss_list.append(self.classificationWd * loss_tensor)
                    self.regularizationCoefficients[v.ref()] = self.classificationWd
        # self.regularizationLoss = tf.add_n(l2_loss_list)

    def build_final_model(self):
        # Build the final loss
        # Temporary model for getting the list of trainable variables
        # self.model = tf.keras.Model(inputs=self.feedDict,
        #                             outputs=[self.evalDict,
        #                                      self.classificationLosses,
        #                                      self.informationGainRoutingLosses])
        self.model = tf.keras.Model(inputs=self.feedDict,
                                    outputs=[self.classificationLosses,
                                             self.informationGainRoutingLosses])

        # self.model2 = tf.keras.Model(inputs=[self.inputs, self.labels, self.routingTemperatures[0]],
        #                              outputs=self.nodeOutputsDict[0])
        # self.model3 = tf.keras.Model(inputs=self.feedDict,
        #                              outputs=self.nodeOutputsDict[0])
        variables = self.model.trainable_variables
        self.calculate_regularization_coefficients(trainable_variables=variables)

    def mask_inputs(self, node):
        f_input, h_input, ig_mask = None, None, None
        if node.isRoot:
            ig_mask = tf.ones_like(self.labels)
            f_input = self.inputs
            self.nodeOutputsDict[node.index]["labels"] = self.labels
            self.nodeOutputsDict[node.index]["batch_indices"] = self.batchIndices
            self.evalDict[Utilities.get_variable_name(name="batch_indices", node=node)] = self.batchIndices
            self.evalDict[Utilities.get_variable_name(name="parent_ig_mask", node=node)] = ig_mask
            self.evalDict[Utilities.get_variable_name(name="parent_sc_mask", node=node)] = ig_mask
            self.evalDict[Utilities.get_variable_name(name="sample_count", node=node)] = tf.shape(self.batchIndices)[0]
        else:
            # Obtain the mask vectors
            parent_node = self.dagObject.parents(node=node)[0]
            sibling_index = self.get_node_sibling_index(node=node)
            parent_ig_mask_matrix = self.nodeOutputsDict[parent_node.index]["ig_mask_matrix"]
            parent_secondary_mask_matrix = self.nodeOutputsDict[parent_node.index]["secondary_mask_matrix"]
            parent_labels = self.nodeOutputsDict[parent_node.index]["labels"]
            parent_batch_indices = self.nodeOutputsDict[parent_node.index]["batch_indices"]
            parent_F = self.nodeOutputsDict[parent_node.index]["F"]
            parent_H = self.nodeOutputsDict[parent_node.index]["H"]
            parent_outputs = [parent_ig_mask_matrix,
                              parent_secondary_mask_matrix,
                              parent_labels,
                              parent_batch_indices,
                              parent_F,
                              parent_H]
            with tf.control_dependencies(parent_outputs):
                # Information gain mask and the secondary routing mask
                parent_ig_mask = parent_ig_mask_matrix[:, sibling_index]
                parent_sc_mask = parent_secondary_mask_matrix[:, sibling_index]
                self.evalDict[Utilities.get_variable_name(name="parent_ig_mask", node=node)] = parent_ig_mask
                self.evalDict[Utilities.get_variable_name(name="parent_sc_mask", node=node)] = parent_sc_mask
                # Mask all required data from the parent: USE SECONDARY MASK
                ig_mask = tf.boolean_mask(parent_ig_mask, parent_sc_mask)
                labels = tf.boolean_mask(parent_labels, parent_sc_mask)
                batch_indices = tf.boolean_mask(parent_batch_indices, parent_sc_mask)
                self.evalDict[Utilities.get_variable_name(name="batch_indices", node=node)] = batch_indices
                f_input = tf.boolean_mask(parent_F, parent_sc_mask)
                h_input = tf.boolean_mask(parent_H, parent_sc_mask)
                # Some intermediate statistics and calculations
                sample_count_tensor = tf.reduce_sum(tf.cast(parent_sc_mask, tf.float32))
                is_node_open = tf.greater_equal(sample_count_tensor, 0.0)
                self.evalDict[Utilities.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
                self.evalDict[Utilities.get_variable_name(name="is_open", node=node)] = is_node_open
                self.nodeOutputsDict[node.index]["labels"] = labels
                self.nodeOutputsDict[node.index]["batch_indices"] = batch_indices
        return f_input, h_input, ig_mask

    def apply_decision(self, node, ig_mask, h_input=None):
        h_net = self.nodeOutputsDict[node.index]["H"]
        labels = self.nodeOutputsDict[node.index]["labels"]
        node_degree = self.degreeList[node.depth]
        # Calculate routing probabilities
        h_ig_net = tf.boolean_mask(h_net, ig_mask)
        h_net_normed = self.maskedBatchNormOps[node.index]([h_net, h_ig_net])
        activations = Cign.fc_layer(x=h_net_normed,
                                    output_dim=node_degree,
                                    activation=None,
                                    node=node,
                                    name="fc_op_decision",
                                    use_bias=True)
        # Routing temperatures
        activations_with_temperature = activations / self.routingTemperatures[node.index]
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_c_given_x = tf.one_hot(labels, self.classCount)
        p_n_given_x_masked = tf.boolean_mask(p_n_given_x, ig_mask)
        p_c_given_x_masked = tf.boolean_mask(p_c_given_x, ig_mask)
        self.evalDict[Utilities.get_variable_name(name="p_n_given_x", node=node)] = p_n_given_x
        self.evalDict[Utilities.get_variable_name(name="p_c_given_x", node=node)] = p_c_given_x
        self.evalDict[Utilities.get_variable_name(name="p_n_given_x_masked", node=node)] = p_n_given_x_masked
        self.evalDict[Utilities.get_variable_name(name="p_c_given_x_masked", node=node)] = p_c_given_x_masked
        # Information gain loss
        information_gain = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x_masked,
                                                 p_c_given_x_2d=p_c_given_x_masked,
                                                 balance_coefficient=self.informationGainBalanceCoeff)
        self.informationGainRoutingLosses[node.index] = information_gain
        self.evalDict[Utilities.get_variable_name(name="information_gain", node=node)] = information_gain
        # Information gain based routing matrix
        ig_routing_matrix = tf.one_hot(tf.argmax(p_n_given_x, axis=1), node_degree, dtype=tf.int32)
        self.evalDict[Utilities.get_variable_name(name="ig_routing_matrix_without_mask", node=node)] = ig_routing_matrix
        mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
        assert "ig_mask_matrix" not in self.nodeOutputsDict[node.index]
        self.nodeOutputsDict[node.index]["ig_mask_matrix"] = \
            tf.cast(
                tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
                dtype=tf.int32)

    def apply_classification_loss(self, node, f_input=None, sc_mask=None):
        f_net = self.nodeOutputsDict[node.index]["F"]
        labels = self.nodeOutputsDict[node.index]["labels"]
        logits = Cign.fc_layer(x=f_net, output_dim=self.classCount, activation=None, node=node, use_bias=True)
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                                   logits=logits)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.math.is_nan(pre_loss), 0.0, pre_loss)
        self.classificationLosses[node.index] = loss

    def get_level_outputs(self, level):
        nodes = self.orderedNodesPerLevel[level]
        expected_outputs = {"F", "H", "labels", "batch_indices", "ig_mask_matrix"}
        assert all([expected_outputs == set(self.nodeOutputsDict[node.index].keys()) for node in nodes])
        x_outputs = [self.nodeOutputsDict[node.index]["F"] for node in nodes]
        h_outputs = [self.nodeOutputsDict[node.index]["H"] for node in nodes]
        ig_outputs = [self.nodeOutputsDict[node.index]["ig_mask_matrix"] for node in nodes]
        label_outputs = [self.nodeOutputsDict[node.index]["labels"] for node in nodes]
        batch_indices_outputs = [self.nodeOutputsDict[node.index]["batch_indices"] for node in nodes]
        return x_outputs, h_outputs, ig_outputs, label_outputs, batch_indices_outputs

    def calculate_secondary_routing_matrix(self, input_f_tensor, input_ig_routing_matrix):
        secondary_routing_matrix = tf.identity(input_ig_routing_matrix)
        return secondary_routing_matrix

    def build_secondary_routing_matrices(self, level):
        x_outputs, h_outputs, ig_outputs, label_outputs, batch_indices_outputs = self.get_level_outputs(level=level)
        # For the vanilla CIGN, secondary routing matrix is equal to the IG one.
        dependencies = []
        dependencies.extend(x_outputs)
        dependencies.extend(h_outputs)
        dependencies.extend(ig_outputs)
        dependencies.extend(label_outputs)
        dependencies.extend(batch_indices_outputs)
        with tf.control_dependencies(dependencies):
            nodes = self.orderedNodesPerLevel[level]
            f_outputs_with_scatter_nd = []
            ig_matrices_with_scatter_nd = []
            for node in nodes:
                batch_size = tf.expand_dims(self.batchSize, axis=0)
                f_output = self.nodeOutputsDict[node.index]["F"]
                ig_routing_matrix = self.nodeOutputsDict[node.index]["ig_mask_matrix"]
                batch_indices_vector = self.nodeOutputsDict[node.index]["batch_indices"]
                # F output
                f_output_shape = tf.shape(f_output)[1:]
                f_scatter_nd_shape = tf.concat([batch_size, f_output_shape], axis=0)
                f_scatter_nd_output = tf.scatter_nd(tf.expand_dims(batch_indices_vector, axis=-1), f_output,
                                                    f_scatter_nd_shape)
                f_outputs_with_scatter_nd.append(f_scatter_nd_output)
                # IG routing matrix
                ig_output_shape = tf.shape(ig_routing_matrix)[1:]
                ig_scatter_nd_shape = tf.concat([batch_size, ig_output_shape], axis=0)
                ig_scatter_nd_output = tf.scatter_nd(tf.expand_dims(batch_indices_vector, axis=-1), ig_routing_matrix,
                                                     ig_scatter_nd_shape)
                ig_matrices_with_scatter_nd.append(ig_scatter_nd_output)
            input_f_tensor = tf.concat(f_outputs_with_scatter_nd, axis=-1)
            ig_combined_routing_matrix = tf.concat(ig_matrices_with_scatter_nd, axis=-1)
            sc_combined_routing_matrix_pre_mask = self.calculate_secondary_routing_matrix(
                input_f_tensor=input_f_tensor, input_ig_routing_matrix=ig_combined_routing_matrix)
            self.evalDict["input_f_tensor_{0}".format(level)] = input_f_tensor
            self.evalDict["ig_combined_routing_matrix_level_{0}".format(level)] = ig_combined_routing_matrix
            self.evalDict["sc_combined_routing_matrix_pre_mask_level_{0}".format(level)] = \
                sc_combined_routing_matrix_pre_mask
            sc_combined_routing_matrix = tf.cast(tf.logical_or(
                tf.cast(sc_combined_routing_matrix_pre_mask, dtype=tf.bool),
                tf.cast(ig_combined_routing_matrix, dtype=tf.bool)), dtype=tf.int32)
            self.evalDict["sc_combined_routing_matrix_level_{0}".format(level)] = sc_combined_routing_matrix
            # Distribute the results of the secondary routing matrix into the corresponding nodes
            curr_column = 0
            for node in nodes:
                node_child_count = len(self.dagObject.children(node=node))
                batch_indices_vector = self.nodeOutputsDict[node.index]["batch_indices"]
                sc_routing_matrix_for_node = sc_combined_routing_matrix[:, curr_column: curr_column + node_child_count]
                ig_routing_matrix_for_node = ig_combined_routing_matrix[:, curr_column: curr_column + node_child_count]
                self.evalDict["sc_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                    sc_routing_matrix_for_node
                self.evalDict["ig_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                    ig_routing_matrix_for_node
                self.nodeOutputsDict[node.index]["secondary_mask_matrix"] = \
                    tf.gather_nd(sc_routing_matrix_for_node, tf.expand_dims(batch_indices_vector, axis=-1))
                self.evalDict["ig_routing_matrix_for_node_{0}_reconstruction".format(node.index)] = \
                    tf.gather_nd(ig_routing_matrix_for_node, tf.expand_dims(batch_indices_vector, axis=-1))
                curr_column += node_child_count

    def calculate_total_loss(self, classification_losses, info_gain_losses):
        # Weight decaying
        variables = self.model.trainable_variables
        regularization_losses = []
        for var in variables:
            if var.ref() in self.regularizationCoefficients:
                lambda_coeff = self.regularizationCoefficients[var.ref()]
                regularization_losses.append(lambda_coeff * tf.nn.l2_loss(var))
        total_regularization_loss = tf.add_n(regularization_losses)
        # Classification losses
        classification_loss = tf.add_n([loss for loss in classification_losses.values()])
        # Information Gain losses
        info_gain_loss = tf.add_n([loss for loss in info_gain_losses.values()])
        # Total loss
        total_loss = total_regularization_loss + info_gain_loss + classification_loss
        return total_loss

    def get_feed_dict(self, x, y, iteration):
        self.softmaxDecayController.update(iteration=iteration + 1)
        temp_value = self.softmaxDecayController.get_value()
        # Fill the feed dict.
        feed_dict = {}
        for input_name in self.feedDict.keys():
            if "routingTemperature" in input_name:
                feed_dict[input_name] = temp_value
            elif input_name == "inputs":
                feed_dict[input_name] = x
            elif input_name == "labels":
                feed_dict[input_name] = y
            else:
                raise NotImplementedError()
        return feed_dict

    def train_step(self, x, y, iteration):
        # eval_dict, classification_losses, info_gain_losses = self.model(inputs=self.feedDict, training=True)
        with tf.GradientTape() as tape:
            t0 = time.time()
            feed_dict = self.get_feed_dict(x=x, y=y, iteration=iteration)
            t1 = time.time()
            classification_losses, info_gain_losses = self.model(inputs=feed_dict, training=True)
            t2 = time.time()
            total_loss = self.calculate_total_loss(classification_losses=classification_losses,
                                                   info_gain_losses=info_gain_losses)
            t3 = time.time()
            # self.unit_test_cign_routing_mechanism(eval_dict=eval_dict)
            t4 = time.time()
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        t5 = time.time()
        print("total={0} [get_feed_dict]t1-t0={1} [self.model]t2-t1={2} [calculate_total_loss]t3-t2={3}"
              " [unit_test_cign_routing_mechanism]t4-t3={4} [tape.gradient]t5-t4={5}".
              format(t5-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))

        # with tf.GradientTape() as tape:
        #     # Get softmax decay value
        #     self.softmaxDecayController.update(iteration=iteration + 1)
        #     temp_value = self.softmaxDecayController.get_value()
        #     # Fill the feed dict.
        #     feed_dict = {}
        #     for input_name in self.feedDict.keys():
        #         if "routingTemperature" in input_name:
        #             feed_dict[input_name] = temp_value
        #         elif input_name == "inputs":
        #             feed_dict[input_name] = x
        #         elif input_name == "labels":
        #             feed_dict[input_name] = y
        #         else:
        #             raise NotImplementedError()
        #     print("X")
        #     eval_dict, classification_losses, info_gain_losses = self.model(inputs=self.feedDict, training=True)
        #     total_loss = self.calculate_total_loss(classification_losses=classification_losses,
        #                                            info_gain_losses=info_gain_losses)
        # grads = tape.gradient(total_loss, self.model.trainable_variables)
        # print("X")
        # self.optimizer.apply_gradients(zip(grads, self.detectorModel.trainable_variables))

    def train(self, dataset, epoch_count):
        iteration = 0
        for epoch_id in range(epoch_count):
            for train_X, train_y in dataset.trainDataTf:
                self.train_step(x=train_X, y=train_y, iteration=iteration)
                iteration += 1

    def unit_test_cign_routing_mechanism(self, eval_dict):
        # Statistics of sample distribution
        for node in self.topologicalSortedNodes:
            key_name = Utilities.get_variable_name(name="sample_count", node=node)
            if key_name in eval_dict:
                print("Node{0} Sample count:{1}".format(node.index, eval_dict[key_name].numpy()))
        # Check if boolean mask works as intended.
        for node in self.topologicalSortedNodes:
            assert Utilities.get_variable_name(name="parent_ig_mask", node=node) in eval_dict
            # Assertion of ig_routing_matrix is correctly derived from routing probabilities.
            if not node.isLeaf:
                p_n_given_x = eval_dict[Utilities.get_variable_name(name="p_n_given_x", node=node)].numpy()
                ig_routing_matrix_without_mask = eval_dict[
                    Utilities.get_variable_name(name="ig_routing_matrix_without_mask", node=node)].numpy()
                ig_routing_matrix_without_mask_np = np.zeros_like(p_n_given_x,
                                                                  dtype=ig_routing_matrix_without_mask.dtype)
                ig_routing_matrix_without_mask_np[np.arange(ig_routing_matrix_without_mask_np.shape[0]),
                                                  np.argmax(p_n_given_x, axis=1)] = 1
                assert np.array_equal(ig_routing_matrix_without_mask, ig_routing_matrix_without_mask_np)
            # Assertion of batch_indices are correctly derived from the parent
            if not node.isRoot:
                parent_node = self.dagObject.parents(node=node)[0]
                node_batch_indices = eval_dict[Utilities.get_variable_name(name="batch_indices", node=node)].numpy()
                parent_batch_indices = eval_dict[
                    Utilities.get_variable_name(name="batch_indices", node=parent_node)].numpy()
                sc_routing_mask = eval_dict[Utilities.get_variable_name(name="parent_sc_mask", node=node)].numpy()
                node_batch_indices_np = parent_batch_indices[sc_routing_mask.astype(np.bool)]
                assert np.array_equal(node_batch_indices, node_batch_indices_np)
        # Check scatter - gather operation correctness for the secondary routing matrix
        batch_size = eval_dict["batch_size"].numpy()
        for level in range(len(self.orderedNodesPerLevel) - 1):
            f_outputs = []
            ig_matrices = []
            for node in self.orderedNodesPerLevel[level]:
                batch_indices = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("batch_indices"),
                                                                      node=node)].numpy()
                f_output = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("F"), node=node)].numpy()
                ig_mask_matrix = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("ig_mask_matrix"),
                                                                       node=node)].numpy()
                f_output_full = np.zeros(shape=[batch_size, *f_output.shape[1:]], dtype=f_output.dtype)
                f_output_full[batch_indices, :] = f_output
                ig_matrix_full = np.zeros(shape=[batch_size, *ig_mask_matrix.shape[1:]], dtype=ig_mask_matrix.dtype)
                ig_matrix_full[batch_indices, :] = ig_mask_matrix
                f_outputs.append(f_output_full)
                ig_matrices.append(ig_matrix_full)
            # Check the correctness of the scatter_nd operation
            f_outputs_np = np.concatenate(f_outputs, axis=-1)
            ig_matrix_np = np.concatenate(ig_matrices, axis=-1)
            f_outputs_tf = eval_dict["input_f_tensor_{0}".format(level)].numpy()
            ig_matrix_tf = eval_dict["ig_combined_routing_matrix_level_{0}".format(level)].numpy()
            assert np.array_equal(f_outputs_np, f_outputs_tf)
            assert np.array_equal(ig_matrix_np, ig_matrix_tf)
            # Check the correctness of the gather_nd operation
            sc_combined_routing_matrix = eval_dict["sc_combined_routing_matrix_level_{0}".format(level)].numpy()
            ig_combined_routing_matrix = eval_dict["ig_combined_routing_matrix_level_{0}".format(level)].numpy()
            curr_column = 0
            for node in self.orderedNodesPerLevel[level]:
                batch_indices = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("batch_indices"),
                                                                      node=node)].numpy()
                node_child_count = len(self.dagObject.children(node=node))
                sc_routing_matrix_for_node = sc_combined_routing_matrix[:, curr_column: curr_column + node_child_count]
                ig_routing_matrix_for_node = ig_combined_routing_matrix[:, curr_column: curr_column + node_child_count]
                sc_routing_matrix_for_node_masked_np = sc_routing_matrix_for_node[batch_indices]
                ig_routing_matrix_for_node_masked_np = ig_routing_matrix_for_node[batch_indices]
                sc_routing_matrix_for_node_masked_tf = eval_dict[
                    Utilities.get_variable_name(name="node_output_{0}".format("secondary_mask_matrix"),
                                                node=node)].numpy()
                ig_routing_matrix_for_node_masked_tf = eval_dict[
                    Utilities.get_variable_name(name="node_output_{0}".format("ig_mask_matrix"),
                                                node=node)].numpy()
                ig_routing_matrix_for_node_masked_tf_2 = eval_dict[
                    "ig_routing_matrix_for_node_{0}_reconstruction".format(node.index)].numpy()
                assert np.array_equal(sc_routing_matrix_for_node_masked_np, sc_routing_matrix_for_node_masked_tf)
                assert np.array_equal(ig_routing_matrix_for_node_masked_np, ig_routing_matrix_for_node_masked_tf)
                assert np.array_equal(ig_routing_matrix_for_node_masked_np, ig_routing_matrix_for_node_masked_tf_2)
                curr_column += node_child_count
        # Check that every row of the concatenated ig routing matrices for each level sums up to exactly 1.
        for level in range(len(self.orderedNodesPerLevel) - 1):
            sc_combined_routing_matrix = eval_dict["sc_combined_routing_matrix_level_{0}".format(level)].numpy()
            ig_combined_routing_matrix = eval_dict["ig_combined_routing_matrix_level_{0}".format(level)].numpy()
            assert len(sc_combined_routing_matrix.shape) == 2
            assert len(ig_combined_routing_matrix.shape) == 2
            assert sc_combined_routing_matrix.shape[0] == batch_size
            assert ig_combined_routing_matrix.shape[0] == batch_size
            ig_rows_summed = np.sum(ig_combined_routing_matrix, axis=-1)
            assert np.array_equal(ig_rows_summed, np.ones_like(ig_rows_summed))
        print("X")
