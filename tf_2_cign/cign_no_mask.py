from collections import deque
from collections import Counter
import numpy as np
import tensorflow as tf
from algorithms.info_gain import InfoGainLoss
from auxillary.dag_utilities import Dag
from simple_tf.uncategorized.node import Node
from tf_2_cign.cign import Cign
from tf_2_cign.custom_layers.cign_secondary_routing_preparation_layer import CignScRoutingPrepLayer
from tf_2_cign.custom_layers.cign_vanilla_sc_routing_layer import CignVanillaScRoutingLayer
from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.utilities import Utilities
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.cign_masking_layer import CignMaskingLayer
from tf_2_cign.custom_layers.cign_decision_layer import CignDecisionLayer
from tf_2_cign.custom_layers.cign_classification_layer import CignClassificationLayer
import time


# tf.autograph.set_verbosity(10, True)


class CignNoMask(Cign):
    def __init__(self, input_dims, class_count, node_degrees, decision_drop_probability,
                 classification_drop_probability, decision_wd, classification_wd, information_gain_balance_coeff,
                 softmax_decay_controller, bn_momentum=0.9):
        super().__init__(input_dims, class_count, node_degrees, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd,
                         information_gain_balance_coeff, softmax_decay_controller, bn_momentum)
        self.weightedBatchNormOps = {}
        self.maskingLayers = {}
        self.decisionLayers = {}
        self.leafLossLayers = {}
        self.nodeFunctions = {}
        self.isRootDict = {}
        self.igMasksDict = {}
        self.scMasksDict = {}
        self.scRoutingPreparationLayers = []
        self.scRoutingCalculationLayers = []

    def node_build_func(self, node):
        pass

    # We don't apply actual masking anymore. We can remove this layer later.
    # Record corresponding ig and sc masks here.
    def mask_inputs(self, node):
        self.maskingLayers[node.index] = CignMaskingLayer(network=self, node=node)

        if node.isRoot:
            parent_ig_matrix = tf.expand_dims(tf.ones_like(self.labels), axis=1)
            parent_sc_matrix = tf.expand_dims(tf.ones_like(self.labels), axis=1)
            parent_F = self.inputs
            parent_H = self.inputs
        else:
            parent_node = self.dagObject.parents(node=node)[0]
            parent_ig_matrix = self.nodeOutputsDict[parent_node.index]["ig_mask_matrix"]
            parent_sc_matrix = self.nodeOutputsDict[parent_node.index]["secondary_mask_matrix"]
            parent_F = self.nodeOutputsDict[parent_node.index]["F"]
            parent_H = self.nodeOutputsDict[parent_node.index]["H"]

        f_net, h_net, ig_mask, sc_mask, sample_count, is_node_open = \
            self.maskingLayers[node.index]([parent_F, parent_H, parent_ig_matrix, parent_sc_matrix])

        self.igMasksDict[node.index] = ig_mask
        self.scMasksDict[node.index] = sc_mask

        self.evalDict[Utilities.get_variable_name(name="f_net", node=node)] = f_net
        self.evalDict[Utilities.get_variable_name(name="h_net", node=node)] = h_net
        self.evalDict[Utilities.get_variable_name(name="ig_mask", node=node)] = ig_mask
        self.evalDict[Utilities.get_variable_name(name="sc_mask", node=node)] = sc_mask
        self.evalDict[Utilities.get_variable_name(name="sample_count", node=node)] = sample_count
        self.evalDict[Utilities.get_variable_name(name="is_node_open", node=node)] = is_node_open
        return f_net, h_net, ig_mask, sc_mask

    # Provide F and H outputs for a given node.
    def apply_node_funcs(self, node, f_net, h_net, ig_mask, sc_mask):
        node_func = self.node_build_func(node=node)
        self.nodeFunctions[node.index] = node_func
        self.nodeOutputsDict[node.index]["H"] = None
        if not node.isLeaf:
            f_net, h_net, pre_branch_feature = self.nodeFunctions[node.index]([f_net, h_net, ig_mask, sc_mask])
            self.evalDict[Utilities.get_variable_name(name="h_net", node=node)] = h_net
            self.evalDict[Utilities.get_variable_name(name="pre_branch_feature", node=node)] = pre_branch_feature
            self.nodeOutputsDict[node.index]["H"] = h_net
        else:
            f_net = self.nodeFunctions[node.index]([f_net, h_net, ig_mask, sc_mask])
        self.evalDict[Utilities.get_variable_name(name="f_net", node=node)] = f_net
        self.nodeOutputsDict[node.index]["F"] = f_net
        return f_net, h_net

    # Provide ig_mask_matrix output for a given inner node and calculate the information gain loss.
    def apply_decision(self, node, ig_mask, h_net=None):
        assert h_net is not None
        self.decisionLayers[node.index] = CignDecisionLayer(network=self, node=node,
                                                            decision_bn_momentum=self.bnMomentum)
        labels = self.labels
        temperature = self.routingTemperatures[node.index]

        h_net_normed, ig_value, output_ig_routing_matrix = self.decisionLayers[node.index]([h_net, ig_mask,
                                                                                            labels, temperature])
        self.informationGainRoutingLosses[node.index] = ig_value

        self.evalDict[Utilities.get_variable_name(name="h_net_normed", node=node)] = h_net_normed
        self.evalDict[Utilities.get_variable_name(name="ig_value", node=node)] = ig_value
        self.evalDict[Utilities.get_variable_name(name="output_ig_routing_matrix",
                                                  node=node)] = output_ig_routing_matrix
        self.nodeOutputsDict[node.index]["ig_mask_matrix"] = output_ig_routing_matrix

    # Calculate the cross entropy loss for this leaf node; by paying attention to the secondary mask vector.
    def apply_classification_loss(self, node, f_net=None, sc_mask=None):
        assert f_net is not None and sc_mask is not None
        self.leafLossLayers[node.index] = CignClassificationLayer(network=self, node=node, class_count=self.classCount)
        labels = self.labels

        cross_entropy_loss_tensor, probability_vector, weighted_losses, loss = self.leafLossLayers[node.index](
            [f_net, sc_mask, labels])
        self.classificationLosses[node.index] = loss

        self.evalDict[Utilities.get_variable_name(name="cross_entropy_loss_tensor", node=node)] = \
            cross_entropy_loss_tensor
        self.evalDict[Utilities.get_variable_name(name="probability_vector", node=node)] = probability_vector
        self.evalDict[Utilities.get_variable_name(name="weighted_losses", node=node)] = weighted_losses
        self.evalDict[Utilities.get_variable_name(name="loss", node=node)] = loss

    def calculate_secondary_routing_matrix(self, input_f_tensor, input_ig_routing_matrix):
        sc_routing_calculation_layer = CignVanillaScRoutingLayer(network=self)
        self.scRoutingCalculationLayers.append(sc_routing_calculation_layer)
        return sc_routing_calculation_layer

    def build_secondary_routing_matrices(self, level):
        level_nodes = self.orderedNodesPerLevel[level]
        f_outputs = [self.nodeOutputsDict[node.index]["F"] for node in level_nodes]
        ig_matrices = [self.nodeOutputsDict[node.index]["ig_mask_matrix"] for node in level_nodes]
        sc_masks = [self.scMasksDict[node.index] for node in level_nodes]
        input_prep_layer = CignScRoutingPrepLayer(network=self, level=level)
        self.scRoutingPreparationLayers.append(input_prep_layer)
        input_f_tensor, input_ig_routing_matrix = input_prep_layer([f_outputs, ig_matrices, sc_masks])
        sc_routing_calculation_layer = self.calculate_secondary_routing_matrix(
            input_f_tensor=input_f_tensor,
            input_ig_routing_matrix=input_ig_routing_matrix)
        secondary_routing_matrix = sc_routing_calculation_layer([input_f_tensor, input_ig_routing_matrix])
        # Distribute the results of the secondary routing matrix into the corresponding nodes
        curr_column = 0
        for node in level_nodes:
            node_child_count = len(self.dagObject.children(node=node))
            sc_routing_matrix_for_node = secondary_routing_matrix[:, curr_column: curr_column + node_child_count]
            self.evalDict["sc_routing_matrix_for_node_{0}_level_{1}".format(node.index, level)] = \
                sc_routing_matrix_for_node
            self.nodeOutputsDict[node.index]["secondary_mask_matrix"] = sc_routing_matrix_for_node
            curr_column += node_child_count

    def build_network(self):
        self.build_tree()
        self.isBaseline = len(self.topologicalSortedNodes) == 1
        # Build all operations in each node -> Level by level
        for level in range(len(self.orderedNodesPerLevel)):
            for node in self.orderedNodesPerLevel[level]:
                print("Building Node:{0}".format(node.index))
                self.isRootDict[node.index] = tf.constant(node.isRoot)
                # Masking Layer
                f_net, h_net, ig_mask, sc_mask = self.mask_inputs(node=node)
                # Actions to be done in the node
                f_net, h_net = \
                    self.apply_node_funcs(node=node, f_net=f_net, h_net=h_net, ig_mask=ig_mask, sc_mask=sc_mask)
                # Decision Layer for inner nodes
                if level < len(self.orderedNodesPerLevel) - 1:
                    self.routingTemperatures[node.index] = \
                        tf.keras.Input(shape=(), name="routingTemperature_{0}".format(node.index))
                    self.feedDict["routingTemperature_{0}".format(node.index)] = self.routingTemperatures[node.index]
                    self.apply_decision(node=node, ig_mask=ig_mask, h_net=h_net)
                else:
                    self.apply_classification_loss(node=node,
                                                   f_net=f_net,
                                                   sc_mask=sc_mask)
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

    def build_final_model(self):
        # Build the final loss
        # Temporary model for getting the list of trainable variables
        self.model = tf.keras.Model(inputs=self.feedDict,
                                    outputs=[self.evalDict,
                                             self.classificationLosses,
                                             self.informationGainRoutingLosses])
        variables = self.model.trainable_variables
        self.calculate_regularization_coefficients(trainable_variables=variables)

