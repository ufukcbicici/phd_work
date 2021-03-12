from tf_2_cign.cign import Cign
import tensorflow as tf
from tf_2_cign.utilities import Utilities


class FashionCign(Cign):
    def __init__(self, input_dims, node_degrees, filter_counts, kernel_sizes, hidden_layers, decision_drop_probability,
                 classification_drop_probability, decision_wd, classification_wd, decision_dimensions,
                 node_build_funcs):
        super().__init__(input_dims, node_degrees, decision_drop_probability, classification_drop_probability,
                         decision_wd, classification_wd)
        self.filterCounts = filter_counts
        self.kernelSizes = kernel_sizes
        self.hiddenLayers = hidden_layers
        self.decisionDimensions = decision_dimensions
        self.nodeBuildFuncs = node_build_funcs

    def apply_router_transformation(self, net, node, decision_feature_size):
        h_net = net
        self.evalDict[Utilities.get_variable_name(name="pre_branch_feature", node=node)] = h_net
        h_net = tf.keras.layers.GlobalAveragePooling2D()(h_net)
        h_net = Cign.fc_layer(x=h_net,
                              output_dim=decision_feature_size,
                              activation="relu",
                              node=node,
                              use_bias=True)
        h_net = tf.keras.layers.Dropout(rate=self.decisionDropProbability)(h_net)
        return h_net

    def root_func(self, node, f_input, h_input):
        num_of_filters = self.filterCounts[0]
        kernel_size = self.kernelSizes[0]
        # F ops
        net = f_input
        net = Cign.conv_layer(x=net,
                              kernel_size=kernel_size,
                              num_of_filters=num_of_filters,
                              strides=(1, 1),
                              node=node,
                              activation="relu",
                              use_bias=True,
                              padding="same")
        net = tf.keras.layers.MaxPooling2D(2)(net)
        # F ops

        # H ops
        decision_dim = self.decisionDimensions[node.depth]
        h_net = self.apply_router_transformation(net=net, node=node, decision_feature_size=decision_dim)
        # H ops

        # Output
        self.nodeOutputsDict[node.index]["F"] = net
        self.nodeOutputsDict[node.index]["H"] = h_net
