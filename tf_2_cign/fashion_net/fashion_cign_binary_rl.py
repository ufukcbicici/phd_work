import tensorflow as tf
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.cign_rl_routing_with_iterative_training import CignRlRoutingWithIterativeTraining
from tf_2_cign.cign_rl_binary_routing import CignRlBinaryRouting
from tf_2_cign.custom_layers.cign_conv_dense_q_net import CignConvDenseQNet
from tf_2_cign.custom_layers.fashion_net_layers.fashion_net_inner_node_func import FashionNetInnerNodeFunc
from tf_2_cign.custom_layers.fashion_net_layers.fashion_net_leaf_node_func import FashionNetLeafNodeFunc


class FashionRlBinaryRouting(CignRlBinaryRouting):
    def __init__(self,
                 valid_prediction_reward,
                 invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 lambda_mac_cost,
                 q_net_params,
                 warm_up_period,
                 cign_rl_train_period,
                 batch_size,
                 input_dims,
                 node_degrees,
                 filter_counts,
                 kernel_sizes,
                 hidden_layers,
                 decision_drop_probability,
                 classification_drop_probability,
                 decision_wd,
                 classification_wd,
                 decision_dimensions,
                 class_count,
                 information_gain_balance_coeff,
                 softmax_decay_controller,
                 learning_rate_schedule,
                 decision_loss_coeff,
                 q_net_coeff,
                 epsilon_decay_rate,
                 epsilon_step,
                 reward_type):
        super().__init__(valid_prediction_reward,
                         invalid_prediction_penalty,
                         include_ig_in_reward_calculations,
                         lambda_mac_cost,
                         warm_up_period,
                         cign_rl_train_period,
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
                         q_net_coeff=1.0,
                         epsilon_decay_rate=epsilon_decay_rate,
                         epsilon_step=epsilon_step,
                         reward_type=reward_type
                         )
        self.filterCounts = filter_counts
        self.kernelSizes = kernel_sizes
        self.hiddenLayers = hidden_layers
        self.decisionDimensions = decision_dimensions
        self.qNetParams = q_net_params

    def node_build_func(self, node):
        if not node.isLeaf:
            node_func = FashionNetInnerNodeFunc(node=node,
                                                network=self,
                                                kernel_size=self.kernelSizes[node.depth],
                                                num_of_filters=self.filterCounts[node.depth],
                                                strides=(1, 1),
                                                activation="relu",
                                                decision_dim=self.decisionDimensions[node.depth],
                                                decision_drop_probability=self.decisionDropProbability,
                                                use_bias=True)
        else:
            node_func = FashionNetLeafNodeFunc(node=node,
                                               network=self,
                                               kernel_size=self.kernelSizes[node.depth],
                                               num_of_filters=self.filterCounts[node.depth],
                                               strides=(1, 1),
                                               activation="relu",
                                               hidden_layer_dims=self.hiddenLayers,
                                               classification_dropout_prob=self.classificationDropProbability,
                                               use_bias=True)

        return node_func

    def get_explanation_string(self):
        explanation = "Fashion CICN with BINARY RL Routing\n"
        explanation += super().get_explanation_string()
        # Fashion CIGN parameters
        explanation += "filterCounts:{0}\n".format(self.filterCounts)
        explanation += "kernelSizes:{0}\n".format(self.kernelSizes)
        explanation += "hiddenLayers:{0}\n".format(self.hiddenLayers)
        explanation += "decisionDimensions:{0}\n".format(self.decisionDimensions)
        # RL Parameters
        explanation += "Q Net Parameters\n"
        for level, q_net_params in enumerate(self.qNetParams):
            explanation += "Level:{0}\n".format(level)
            explanation += "Level:{0} Q Net Kernel Size:{1}\n".format(level, q_net_params["Conv_Filter"])
            explanation += "Level:{0} Q Net Kernel Strides:{1}\n".format(level, q_net_params["Conv_Strides"])
            explanation += "Level:{0} Q Net Feature Maps:{1}\n".format(level, q_net_params["Conv_Feature_Maps"])
            explanation += "Level:{0} Q Net Hidden Layers:{1}\n".format(level, q_net_params["Hidden_Layers"])
        explanation += "train_period:{0}\n".format(self.cignRlTrainPeriod)
        explanation += "qNetCoeff:{0}\n".format(self.qNetCoeff)
        return explanation

    def get_q_net_layer(self, level):
        node = self.orderedNodesPerLevel[level][-1]
        q_net_params = self.qNetParams[level]
        q_net_layer = CignConvDenseQNet(level=level,
                                        node=node,
                                        network=self,
                                        kernel_size=q_net_params["Conv_Filter"],
                                        num_of_filters=q_net_params["Conv_Feature_Maps"],
                                        strides=q_net_params["Conv_Strides"],
                                        activation="relu",
                                        hidden_layer_dims=q_net_params["Hidden_Layers"],
                                        q_network_dim=2,  # Always binary actions!!!
                                        rl_dropout_prob=self.classificationDropProbability)
        return q_net_layer
