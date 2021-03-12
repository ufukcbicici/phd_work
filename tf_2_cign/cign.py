from collections import deque
from collections import Counter
import numpy as np
import tensorflow as tf

from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from simple_tf.uncategorized.node import Node
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.utilities import Utilities


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
        # Classification losses
        self.classificationLossObjects = {}
        self.classificationLosses = {}
        # Routing Losses
        self.informationGainRoutingLosses = {}
        # Model
        self.model = None

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

    # REVISION OK
    @staticmethod
    def conv_layer(x, kernel_size, num_of_filters, strides, node, activation,
                   use_bias=True, padding="same", name="conv_op"):
        assert len(x.get_shape().as_list()) == 4
        assert strides[0] == strides[1]
        # shape = [filter_size, filter_size, in_filters, out_filters]
        num_of_input_channels = x.get_shape().as_list()[3]
        height_of_input_map = x.get_shape().as_list()[2]
        width_of_input_map = x.get_shape().as_list()[1]
        height_of_filter = kernel_size
        width_of_filter = kernel_size
        num_of_output_channels = num_of_filters
        convolution_stride = strides[0]
        cost = Utilities.calculate_mac_of_computation(
            num_of_input_channels=num_of_input_channels,
            height_of_input_map=height_of_input_map, width_of_input_map=width_of_input_map,
            height_of_filter=height_of_filter, width_of_filter=width_of_filter,
            num_of_output_channels=num_of_output_channels, convolution_stride=convolution_stride
        )
        if node is not None:
            node.macCost += cost
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            node.opMacCostsDict["{0}_{1}".format(name, op_id)] = cost
        # Apply operation
        net = tf.keras.layers.Conv2D(filters=num_of_filters,
                                     kernel_size=kernel_size,
                                     activation=activation,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=use_bias,
                                     name=Utilities.get_variable_name(name="ConvLayer", node=node))(x)
        return net

    # REVISION OK
    @staticmethod
    def fc_layer(x, output_dim, activation, node, use_bias=True, name="fc_op"):
        assert len(x.get_shape().as_list()) == 2
        num_of_input_channels = x.get_shape().as_list()[1]
        num_of_output_channels = output_dim
        cost = Utilities.calculate_mac_of_computation(num_of_input_channels=num_of_input_channels,
                                                      height_of_input_map=1,
                                                      width_of_input_map=1,
                                                      height_of_filter=1,
                                                      width_of_filter=1,
                                                      num_of_output_channels=num_of_output_channels,
                                                      convolution_stride=1,
                                                      type="fc")
        if node is not None:
            node.macCost += cost
            op_id = 0
            while True:
                if "{0}_{1}".format(name, op_id) in node.opMacCostsDict:
                    op_id += 1
                    continue
                break
            node.opMacCostsDict["{0}_{1}".format(name, op_id)] = cost

        # Apply operation
        net = tf.keras.layers.Dense(units=output_dim,
                                    activation=activation,
                                    use_bias=use_bias,
                                    name=Utilities.get_variable_name(name="DenseLayer", node=node))(x)
        return net

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
                    self.apply_decision(node=node, ig_mask=ig_mask)
                else:
                    self.apply_classification_loss(node=node)
            # Build secondary routing matrices after a level's nodes being built (if not the final layer)
            if level < len(self.orderedNodesPerLevel) - 1:
                self.build_secondary_routing_matrices(level=level)
        # Build the model
        # Build the final loss

    def mask_inputs(self, node):
        f_input, h_input, ig_mask = None, None, None
        if node.isRoot:
            ig_mask = tf.ones_like(self.labels)
            f_input = self.inputs
            self.nodeOutputsDict[node.index]["labels"] = self.labels
            self.nodeOutputsDict[node.index]["batch_indices"] = self.batchIndices
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
                # Mask all required data from the parent: USE SECONDARY MASK
                ig_mask = tf.boolean_mask(parent_ig_mask, parent_sc_mask)
                labels = tf.boolean_mask(parent_labels, parent_sc_mask)
                batch_indices = tf.boolean_mask(parent_batch_indices, parent_sc_mask)
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

    def apply_decision(self, node, ig_mask):
        h_net = self.nodeOutputsDict[node.index]["H"]
        labels = self.nodeOutputsDict[node.index]["labels"]
        node_degree = self.degreeList[node.depth]
        # Calculate routing probabilities
        h_ig_net = tf.boolean_mask(h_net, ig_mask)
        h_net_normed = MaskedBatchNormalization(momentum=self.bnMomentum)([h_net, h_ig_net])
        activations = Cign.fc_layer(x=h_net_normed,
                                    output_dim=node_degree,
                                    activation=None,
                                    node=node,
                                    use_bias=True)
        # Routing temperatures
        self.routingTemperatures[node.index] = \
            tf.keras.Input(shape=(), name="routingTemperature_{0}".format(node.index))
        self.feedDict["routingTemperature_{0}".format(node.index)] = self.routingTemperatures[node.index]
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
        ig_routing_matrix = tf.one_hot(tf.argmax(p_n_given_x, axis=1), node_degree)
        self.evalDict[Utilities.get_variable_name(name="ig_routing_matrix_without_mask", node=node)] = ig_routing_matrix
        mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
        assert "ig_mask_matrix" not in self.nodeOutputsDict[node.index]
        self.nodeOutputsDict[node.index]["ig_mask_matrix"] = tf.logical_and(ig_routing_matrix, mask_as_matrix)

    def apply_classification_loss(self, node):
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
            self.evalDict["ig_combined_routing_matrix_level_{0}".format(level)] = ig_combined_routing_matrix
            self.evalDict["sc_combined_routing_matrix_pre_mask_level_{0}".format(level)] = \
                sc_combined_routing_matrix_pre_mask
            sc_combined_routing_matrix = tf.logical_or(sc_combined_routing_matrix_pre_mask, ig_combined_routing_matrix)
            self.evalDict["sc_combined_routing_matrix_level_{0}".format(level)] = sc_combined_routing_matrix
            # Distribute the results of the secondary routing matrix into the corresponding nodes
            curr_column = 0
            for node in nodes:
                node_child_count = len(self.dagObject.children(node=node))
                batch_indices_vector = self.nodeOutputsDict[node.index]["batch_indices"]
                sc_routing_matrix_for_node = sc_combined_routing_matrix[:, curr_column: curr_column+node_child_count]
                ig_routing_matrix_for_node = ig_combined_routing_matrix[:, curr_column: curr_column+node_child_count]
                self.evalDict["sc_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                    sc_routing_matrix_for_node
                self.evalDict["ig_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                    ig_routing_matrix_for_node
                self.nodeOutputsDict[node.index]["secondary_mask_matrix"] = \
                    tf.gather_nd(sc_routing_matrix_for_node, tf.expand_dims(batch_indices_vector, axis=-1))
                self.evalDict["ig_routing_matrix_for_node_{0}_reconstruction".format(node.index)] = \
                    tf.gather_nd(ig_routing_matrix_for_node, tf.expand_dims(batch_indices_vector, axis=-1))

    # def build_final_loss(self):
    #     # Calculate the weight decay loss on variables
