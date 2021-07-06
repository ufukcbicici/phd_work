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
    def __init__(self,
                 batch_size,
                 input_dims,
                 class_count,
                 node_degrees,
                 decision_drop_probability,
                 classification_drop_probability,
                 decision_wd,
                 classification_wd,
                 information_gain_balance_coeff,
                 softmax_decay_controller,
                 learning_rate_schedule,
                 decision_loss_coeff,
                 bn_momentum=0.9):
        super().__init__(batch_size,
                         input_dims,
                         class_count,
                         node_degrees,
                         decision_drop_probability,
                         classification_drop_probability,
                         decision_wd,
                         classification_wd,
                         information_gain_balance_coeff,
                         softmax_decay_controller,
                         learning_rate_schedule,
                         decision_loss_coeff,
                         bn_momentum)
        self.weightedBatchNormOps = {}
        self.maskingLayers = {}
        self.decisionLayers = {}
        self.leafLossLayers = {}
        self.nodeFunctions = {}
        self.isRootDict = {}
        self.igMasksDict = {}
        self.scMasksDict = {}
        self.igActivationsDict = {}
        self.scRoutingPreparationLayers = []
        self.scRoutingCalculationLayers = []
        self.optimizer = None

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

        h_net_normed, ig_value, output_ig_routing_matrix, ig_activations = self.decisionLayers[node.index](
            [h_net, ig_mask,
             labels,
             temperature])
        self.informationGainRoutingLosses[node.index] = ig_value
        self.igActivationsDict[node.index] = ig_activations

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
                self.evalDict[Utilities.get_variable_name(name="ig_mask_vector", node=node)] = self.igMasksDict[
                    node.index]
                self.evalDict[Utilities.get_variable_name(name="sc_mask_vector", node=node)] = self.scMasksDict[
                    node.index]

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
        info_gain_loss = self.decisionLossCoeff * tf.add_n([loss for loss in info_gain_losses.values()])
        # Total loss
        total_loss = total_regularization_loss + info_gain_loss + classification_loss
        return total_loss, total_regularization_loss, info_gain_loss, classification_loss

    def train_step(self, x, y, iteration):
        # eval_dict, classification_losses, info_gain_losses = self.model(inputs=self.feedDict, training=True)
        with tf.GradientTape() as tape:
            t0 = time.time()
            feed_dict = self.get_feed_dict(x=x, y=y, iteration=iteration)
            t1 = time.time()
            eval_dict, classification_losses, info_gain_losses = self.model(inputs=feed_dict, training=True)
            t2 = time.time()
            total_loss, total_regularization_loss, info_gain_loss, classification_loss = self.calculate_total_loss(
                classification_losses=classification_losses,
                info_gain_losses=info_gain_losses)
            t3 = time.time()
            # self.unit_test_cign_routing_mechanism(eval_dict=eval_dict)
            t4 = time.time()
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        t5 = time.time()
        print("total={0} [get_feed_dict]t1-t0={1} [self.model]t2-t1={2} [calculate_total_loss]t3-t2={3}"
              " [unit_test_cign_routing_mechanism]t4-t3={4} [tape.gradient]t5-t4={5}".
              format(t5 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4))

    def train(self, dataset, epoch_count):
        boundaries = [tpl[0] for tpl in self.learningRateSchedule.schedule]
        values = [self.learningRateSchedule.initialValue]
        values.extend([tpl[1] for tpl in self.learningRateSchedule.schedule])
        learning_rate_scheduler_tf = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_scheduler_tf, momentum=0.9)

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

        # Assert that all mask vectors sum up to unity at each row in the leaf layer.
        ig_masks = []
        for node in self.orderedNodesPerLevel[-1]:
            ig_masks.append(eval_dict[Utilities.get_variable_name(name="ig_mask_vector", node=node)])
        ig_matrix = np.stack(ig_masks, axis=1)
        row_sums = np.sum(ig_matrix, axis=1)
        assert np.array_equal(row_sums, np.ones_like(row_sums))

        # Assert that all ig routing masks of the child nodes; when concatenated are equal to the parent's ig routing
        # matrix.
        for node in self.topologicalSortedNodes:
            if node.isLeaf:
                continue
            ig_matrix = eval_dict[Utilities.get_variable_name(name="node_output_{0}".format("ig_mask_matrix"),
                                                              node=node)]
            child_nodes = self.dagObject.children(node=node)
            ig_masks_of_children = [None] * len(child_nodes)
            for child_node in child_nodes:
                sibling_index = self.get_node_sibling_index(node=child_node)
                ig_masks_of_children[sibling_index] = eval_dict[Utilities.get_variable_name(name="ig_mask_vector",
                                                                                            node=child_node)]
            ig_matrix_constructed = np.stack(ig_masks_of_children, axis=1)
            assert np.array_equal(ig_matrix, ig_matrix_constructed)

        print("All assertions are correct")

    def get_explanation_string(self):
        explanation = ""
        total_param_count = 0
        for v in self.model.trainable_variables:
            total_param_count += np.prod(v.get_shape().as_list())

        explanation += "Total Param Count:{0}\n".format(total_param_count)
        explanation += "Batch Size:{0}\n".format(self.batchSize)
        explanation += "Tree Degree:{0}\n".format(self.degreeList)
        explanation += "********Lr Settings********\n"
        explanation += self.learningRateSchedule.get_explanation()
        explanation += "********Lr Settings********\n"
        explanation += "Decision Loss Coeff:{0}\n".format(self.decisionLossCoeff)
        explanation += "Batch Norm Decay:{0}\n".format(self.bnMomentum)
        explanation += "Param Count:{0}\n".format(total_param_count)
        explanation += "Classification Wd:{0}\n".format(self.classificationWd)
        explanation += "Decision Wd:{0}\n".format(self.decisionWd)
        explanation += "Information Gain Balance Coefficient:{0}\n".format(self.informationGainBalanceCoeff)
        explanation += "Use Decision Dropout:{0}\n".format(self.decisionDropProbability)
        explanation += "Use Classification Dropout:{0}\n".format(self.classificationDropProbability)
        return explanation



        # explanation += "F Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_1_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_1)
        # explanation += "F Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_2_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_2)
        # explanation += "F Conv3:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_3_SIZE,
        #                                                        GlobalConstants.FASHION_F_NUM_FILTERS_3)
        # explanation += "F FC1:{0} Units\n".format(GlobalConstants.FASHION_F_FC_1)
        # explanation += "F FC2:{0} Units\n".format(GlobalConstants.FASHION_F_FC_2)
        # explanation += "F Residue FC:{0} Units\n".format(GlobalConstants.FASHION_F_RESIDUE)
        # explanation += "Residue Hidden Layer Count:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT)
        # explanation += "Residue Use Dropout:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT)
        # explanation += "H Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_1_SIZE,
        #                                                        GlobalConstants.FASHION_H_NUM_FILTERS_1)
        # explanation += "H Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_2_SIZE,
        #                                                        GlobalConstants.FASHION_H_NUM_FILTERS_2)
        # explanation += "FASHION_NO_H_FROM_F_UNITS_1:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
        # explanation += "FASHION_NO_H_FROM_F_UNITS_2:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
        # explanation += "Use Class Weighting:{0}\n".format(GlobalConstants.USE_CLASS_WEIGHTING)
        # explanation += "Class Weight Running Average:{0}\n".format(GlobalConstants.CLASS_WEIGHT_RUNNING_AVERAGE)
        # explanation += "Zero Label Count Epsilon:{0}\n".format(GlobalConstants.LABEL_EPSILON)
        # explanation += "TRAINING PARAMETERS:\n"
        # explanation += super().get_explanation_string()
