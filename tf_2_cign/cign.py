from collections import deque
from collections import Counter
import numpy as np
import tensorflow as tf

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
        self.classCount = class_count
        self.decisionDropProbability = decision_drop_probability
        self.classificationDropProbability = classification_drop_probability
        self.decisionWd = decision_wd
        self.classificationWd = classification_wd
        self.bnMomentum = bn_momentum
        # Model-wise Tensorflow objects.
        self.inputs = tf.keras.Input(shape=input_dims, name="inputs")
        self.labels = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)
        self.batchSize = tf.keras.Input(shape=(), name="batchSize", dtype=tf.int32)
        self.batchIndices = tf.range(0, self.batchSize, 1)
        # Hyper-parameters
        self.routingSoftmaxDecays = {}
        # Node input-outputs
        self.labelsDict = {}
        self.batchIndicesDict = {}
        self.nodeInputsDict = {}
        self.nodeOutputsDict = {}
        # Routing temperatures
        self.routingTemperatures = {"inputs": self.inputs, "labels": self.labels, "batchSize": self.batchSize}
        # Feed dict
        self.feedDict = {}
        # Information Gain Mask vectors
        self.igMaskVectorsDict = {}
        # Secondary Routing Mask vectors (Heuristic thresholding, Reinforcement Learning, Bayesian Optimization, etc.)
        self.secondaryMaskVectorsDict = {}
        # Global evaluation dictionary
        self.evalDict = {}
        # Node builder functions
        self.nodeBuildFuncs = []
        # Classification losses
        self.classificationLosses = {}
        # Routing Losses
        self.routingLosses = {}

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
        # Build all operations in each node
        for node in self.topologicalSortedNodes:
            print("Building Node {0}".format(node.index))
            if node.isLeaf is False:
                self.routingTemperatures[node.index] = \
                    tf.keras.Input(shape=(), name="routingTemperature_{0}".format(node.index))
                self.feedDict["routingTemperature_{0}".format(node.index)] = self.routingTemperatures[node.index]
            self.mask_inputs(node=node)
            self.nodeBuildFuncs[node.depth](self=self, node=node)
            if node.isLeaf:
                print("Call loss function")
            else:
                pass

    def mask_inputs(self, node):
        if node.isRoot:
            self.nodeInputsDict[node.index] = {"F": self.inputs}
            self.labelsDict[node.index] = self.labels
            self.batchIndicesDict[node.index] = self.batchIndices
        else:
            # Obtain the mask vectors
            parent_node = self.dagObject.parents(node=node)[0]
            sibling_index = self.get_node_sibling_index(node=node)
            with tf.control_dependencies([self.igMaskVectorsDict[parent_node.index],
                                          self.secondaryMaskVectorsDict[parent_node.index]]):
                # Information gain mask and the secondary routing mask
                parent_ig_mask = self.igMaskVectorsDict[parent_node.index][:, sibling_index]
                parent_sc_mask = self.secondaryMaskVectorsDict[parent_node.index][:, sibling_index]
                # Mask all required data from the parent
                ig_mask = tf.boolean_mask(parent_ig_mask, parent_sc_mask)
                labels = tf.boolean_mask(self.labelsDict[parent_node.index], parent_sc_mask)
                batch_indices = tf.boolean_mask(self.batchIndicesDict[parent_node.index], parent_sc_mask)
                F_input = tf.boolean_mask(self.nodeOutputsDict[parent_node.index]["F"], parent_sc_mask)
                H_input = tf.boolean_mask(self.nodeOutputsDict[parent_node.index]["H"], parent_sc_mask)
                sample_count_tensor = tf.reduce_sum(tf.cast(parent_sc_mask, tf.float32))
                self.evalDict[Utilities.get_variable_name(name="sample_count", node=node)] = sample_count_tensor
                is_node_open = tf.greater_equal(sample_count_tensor, 0.0)
                self.evalDict[Utilities.get_variable_name(name="is_open", node=node)] = is_node_open
                self.nodeInputsDict[node.index] = {"F": F_input, "H": H_input, "ig_mask": ig_mask}
                self.labelsDict[node.index] = labels
                self.batchIndicesDict[node.index] = batch_indices

    def apply_decision(self, node):
        h_net = self.nodeOutputsDict[node.index]["H"]
        ig_mask = self.nodeOutputsDict[node.index]["ig_mask"]
        node_degree = self.degreeList[node.depth]

        h_ig_net = tf.boolean_mask(h_net, ig_mask)
        h_net_normed = MaskedBatchNormalization(momentum=self.bnMomentum)([h_net, h_ig_net])
        activations = Cign.fc_layer(x=h_net_normed,
                                    output_dim=node_degree,
                                    activation=None,
                                    node=node,
                                    use_bias=True)
        activations_with_temperature = activations / self.routingTemperatures[node.index]
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_c_given_x = tf.one_hot(self.labelsDict[node.index], self.classCount)
        p_n_given_x_masked = tf.boolean_mask(p_n_given_x, ig_mask)
        p_c_given_x_masked = tf.one_hot(p_c_given_x, ig_mask)
