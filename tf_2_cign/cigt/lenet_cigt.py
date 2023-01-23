import tensorflow as tf
import numpy as np

from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from simple_tf.uncategorized.node import Node
from tf_2_cign.cigt.cigt import Cigt
from tf_2_cign.cigt.custom_layers.lenet_cigt_layers.lenet_cigt_inner_block import LeNetCigtInnerBlock
from tf_2_cign.cigt.custom_layers.lenet_cigt_layers.lenet_cigt_leaf_block import LeNetCigtLeafBlock


class LenetCigt(Cigt):
    def __init__(self, run_id, batch_size, input_dims, filter_counts, kernel_sizes, hidden_layers,
                 decision_drop_probability, classification_drop_probability, decision_wd, classification_wd,
                 evaluation_period, measurement_start, decision_dimensions, class_count, information_gain_balance_coeff,
                 softmax_decay_controller, learning_rate_schedule, optimizer_type, decision_non_linearity,
                 decision_loss_coeff, path_counts, bn_momentum, warm_up_period, routing_strategy_name,
                 use_straight_through, save_model, model_definition, *args, **kwargs):

        super().__init__(run_id, batch_size, input_dims, class_count, path_counts, softmax_decay_controller,
                         learning_rate_schedule, optimizer_type, decision_non_linearity, decision_loss_coeff,
                         routing_strategy_name, use_straight_through, warm_up_period, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd, evaluation_period,
                         measurement_start, save_model, model_definition, *args, **kwargs)
        self.filterCounts = filter_counts
        self.kernelSizes = kernel_sizes
        self.hiddenLayers = hidden_layers
        self.bnMomentum = bn_momentum
        assert len(path_counts) + 1 == len(filter_counts)
        assert len(filter_counts) == len(kernel_sizes)
        self.decisionDimensions = decision_dimensions
        self.informationGainBalanceCoeff = information_gain_balance_coeff
        self.optimizer = None
        self.build_network()
        # self.dummyBlock = None
        # self.calculate_regularization_coefficients()
        self.optimizer = self.get_optimizer()

    def build_network(self):
        self.cigtBlocks = []
        super(LenetCigt, self).build_network()
        curr_node = self.rootNode
        for block_id, path_count in enumerate(self.pathCounts):
            prev_block_path_count = 1 if block_id == 0 else self.pathCounts[block_id - 1]
            if block_id < len(self.pathCounts) - 1:
                block = LeNetCigtInnerBlock(node=curr_node,
                                            kernel_size=self.kernelSizes[block_id],
                                            num_of_filters=self.filterCounts[block_id],
                                            strides=(1, 1),
                                            activation="relu",
                                            use_bias=True,
                                            padding="same",
                                            decision_drop_probability=self.decisionDropProbability,
                                            decision_dim=self.decisionDimensions[block_id],
                                            bn_momentum=self.bnMomentum,
                                            prev_block_path_count=prev_block_path_count,
                                            this_block_path_count=self.pathCounts[block_id],
                                            next_block_path_count=self.pathCounts[block_id + 1],
                                            ig_balance_coefficient=self.informationGainBalanceCoeff,
                                            class_count=self.classCount,
                                            routing_strategy_name=self.routingStrategyName,
                                            use_straight_through=self.useStraightThrough,
                                            decision_non_linearity=self.decisionNonLinearity)
            else:
                block = LeNetCigtLeafBlock(node=curr_node,
                                           kernel_size=self.kernelSizes[block_id],
                                           num_of_filters=self.filterCounts[block_id],
                                           activation="relu",
                                           hidden_layer_dims=self.hiddenLayers,
                                           classification_dropout_prob=self.classificationDropProbability,
                                           use_bias=True,
                                           padding="same",
                                           strides=(1, 1),
                                           class_count=self.classCount,
                                           prev_block_path_count=prev_block_path_count,
                                           this_block_path_count=self.pathCounts[block_id])
            self.cigtBlocks.append(block)
            if curr_node.isLeaf:
                curr_node = self.dagObject.children(node=curr_node)
                assert len(curr_node) == 0
            else:
                curr_node = self.dagObject.children(node=curr_node)
                assert len(curr_node) == 1
                curr_node = curr_node[0]
        # self.build(input_shape=[(self.batchSize, *self.inputDims), (self.batchSize, )])

    def calculate_total_macs(self):
        leaf_node = self.cigtNodes[-1]
        loss_layer_cost = leaf_node.opMacCostsDict[self.cigtBlocks[-1].lossLayer.opName]
        routed_loss_layer_cost = loss_layer_cost / self.pathCounts[-1]
        leaf_node.macCost -= loss_layer_cost
        leaf_node.macCost += routed_loss_layer_cost
        leaf_node.opMacCostsDict[self.cigtBlocks[-1].lossLayer.opName] = routed_loss_layer_cost
        super(LenetCigt, self).calculate_total_macs()

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation += super().get_explanation_string()
        explanation = self.add_explanation(name_of_param="Filter Counts", value=self.filterCounts,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Kernel Sizes", value=self.kernelSizes,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Hidden Layers", value=self.hiddenLayers,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Decision Dimensions", value=self.decisionDimensions,
                                           explanation=explanation, kv_rows=kv_rows)
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

        # RL Parameters
        # explanation += "Q Net Parameters\n"
        # for level, q_net_params in enumerate(self.qNetParams):
        #     explanation += "Level:{0}\n".format(level)
        #     explanation += "Level:{0} Q Net Kernel Size:{1}\n".format(level, q_net_params["Conv_Filter"])
        #     explanation += "Level:{0} Q Net Kernel Strides:{1}\n".format(level, q_net_params["Conv_Strides"])
        #     explanation += "Level:{0} Q Net Feature Maps:{1}\n".format(level, q_net_params["Conv_Feature_Maps"])
        #     explanation += "Level:{0} Q Net Hidden Layers:{1}\n".format(level, q_net_params["Hidden_Layers"])
        # explanation += "train_period:{0}\n".format(self.cignRlTrainPeriod)
        # explanation += "qNetCoeff:{0}\n".format(self.qNetCoeff)
        # return explanation
