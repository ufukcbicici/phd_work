from tf_2_cign.cign import Cign
import tensorflow as tf

from tf_2_cign.cign_no_mask import CignNoMask
from tf_2_cign.cign_rl_routing import CignRlRouting
from tf_2_cign.fashion_net.fashion_net_inner_node_func import FashionNetInnerNodeFunc
from tf_2_cign.fashion_net.fashion_net_leaf_node_func import FashionNetLeafNodeFunc
from tf_2_cign.utilities import Utilities


class FashionCignRl(CignRlRouting):
    def __init__(self,
                 valid_prediction_reward,
                 invalid_prediction_penalty,
                 include_ig_in_reward_calculations,
                 lambda_mac_cost,
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
                 decision_loss_coeff):
        super().__init__(valid_prediction_reward,
                         invalid_prediction_penalty,
                         include_ig_in_reward_calculations,
                         lambda_mac_cost,
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
                         decision_loss_coeff)
        self.filterCounts = filter_counts
        self.kernelSizes = kernel_sizes
        self.hiddenLayers = hidden_layers
        self.decisionDimensions = decision_dimensions

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
        explanation = "Fashion CICN with RL Routing\n"
        explanation += super().get_explanation_string()
        # Fashion CIGN parameters
        explanation += "filterCounts:{0}\n".format(self.filterCounts)
        explanation += "kernelSizes:{0}\n".format(self.kernelSizes)
        explanation += "hiddenLayers:{0}\n".format(self.hiddenLayers)
        explanation += "decisionDimensions:{0}\n".format(self.decisionDimensions)
        return explanation
