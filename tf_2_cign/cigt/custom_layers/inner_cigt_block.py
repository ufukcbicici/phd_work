import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_decision_layer import CigtDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_gumbel_softmax_decision_layer import CigtGumbelSoftmaxDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_random_decision_layer import CigtRandomDecisionLayer
from tf_2_cign.cigt.custom_layers.cigt_route_averaging_layer import CigtRouteAveragingLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer


class InnerCigtBlock(tf.keras.layers.Layer):
    def __init__(self, node, routing_strategy,
                 decision_drop_probability, decision_dim, bn_momentum,
                 next_block_path_count, class_count, ig_balance_coefficient, use_straight_through,
                 decision_non_linearity):
        super().__init__()
        self.node = node
        self.bnMomentum = bn_momentum
        self.routingStrategy = routing_strategy
        self.useStraightThrough = use_straight_through
        self.decisionNonLinearity = decision_non_linearity

        # H operations
        self.cigtRouteAveragingLayer = CigtRouteAveragingLayer()
        self.decisionGAPLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.decisionDim = decision_dim
        self.decisionDropProbability = decision_drop_probability
        self.decisionFcLayer = CignDenseLayer(output_dim=self.decisionDim,
                                              activation="relu",
                                              node=node,
                                              use_bias=True,
                                              name="Lenet_Cigt_Node_{0}_fc_op_decision".format(self.node.index))
        self.decisionDropoutLayer = tf.keras.layers.Dropout(rate=self.decisionDropProbability)
        if self.routingStrategy == "Full_Training":
            self.cigtDecisionLayer = CigtGumbelSoftmaxDecisionLayer(node=node,
                                                                    decision_bn_momentum=self.bnMomentum,
                                                                    next_block_path_count=next_block_path_count,
                                                                    class_count=class_count,
                                                                    ig_balance_coefficient=ig_balance_coefficient,
                                                                    straight_through=self.useStraightThrough,
                                                                    decision_non_linearity=self.decisionNonLinearity)

        elif self.routingStrategy == "Approximate_Training":
            self.cigtDecisionLayer = CigtDecisionLayer(node=node,
                                                       decision_bn_momentum=self.bnMomentum,
                                                       next_block_path_count=next_block_path_count,
                                                       class_count=class_count,
                                                       ig_balance_coefficient=ig_balance_coefficient,
                                                       from_logits=True)
        elif self.routingStrategy == "Random_Routing":
            self.cigtDecisionLayer = CigtRandomDecisionLayer(node=node,
                                                             decision_bn_momentum=self.bnMomentum,
                                                             next_block_path_count=next_block_path_count,
                                                             class_count=class_count,
                                                             ig_balance_coefficient=ig_balance_coefficient)

        else:
            raise NotImplementedError()

    def call(self, inputs, **kwargs):
        pass
