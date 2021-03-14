from tf_2_cign.cign import Cign
import tensorflow as tf
from tf_2_cign.utilities import Utilities


class FashionCign(Cign):
    def __init__(self, input_dims, node_degrees, filter_counts, kernel_sizes, hidden_layers, decision_drop_probability,
                 classification_drop_probability, decision_wd, classification_wd, decision_dimensions, node_build_funcs,
                 class_count, information_gain_balance_coeff):
        super().__init__(input_dims, class_count, node_degrees, decision_drop_probability,
                         classification_drop_probability, decision_wd, classification_wd,
                         information_gain_balance_coeff)
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

    def conv_block(self, node, input_net, kernel_sizes, filter_counts, strides, pool_dims):
        length_set = {len(kernel_sizes), len(filter_counts), len(strides), len(pool_dims)}
        assert len(length_set) == 1
        layer_count = list(length_set)[0]

        net = input_net
        for layer_id in range(layer_count):
            kernel_size = kernel_sizes[layer_id]
            num_of_filters = filter_counts[layer_id]
            stride = strides[layer_id]
            pool_size = pool_dims[layer_id]
            net = Cign.conv_layer(x=net,
                                  kernel_size=kernel_size,
                                  num_of_filters=num_of_filters,
                                  strides=stride,
                                  node=node,
                                  activation="relu",
                                  use_bias=True,
                                  padding="same")
            net = tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same")(net)
        return net

    def dense_block(self, node, input_net, dims, use_dropout):
        net = input_net
        for layer_id, hidden_dim in enumerate(dims):
            net = Cign.fc_layer(x=net, output_dim=hidden_dim, activation="relu", node=node)
            if use_dropout:
                net = tf.keras.layers.Dropout(rate=self.classificationDropProbability)(net)
        return net

    def inner_func(self, node, f_input, h_input):
        num_of_filters = self.filterCounts[node.depth]
        kernel_size = self.kernelSizes[node.depth]
        # F ops
        net = self.conv_block(node=node,
                              input_net=f_input,
                              kernel_sizes=[kernel_size],
                              filter_counts=[num_of_filters],
                              strides=[(1, 1)],
                              pool_dims=[2])
        # F ops

        # H ops
        decision_dim = self.decisionDimensions[node.depth]
        h_net = self.apply_router_transformation(net=net, node=node, decision_feature_size=decision_dim)
        # H ops

        # Output
        self.nodeOutputsDict[node.index]["F"] = net
        self.nodeOutputsDict[node.index]["H"] = h_net

    def leaf_func(self, node, f_input, h_input):
        num_of_filters = self.filterCounts[node.depth]
        kernel_size = self.kernelSizes[node.depth]
        # F ops
        # 1 Conv layer
        net = self.conv_block(node=node,
                              input_net=f_input,
                              kernel_sizes=[kernel_size],
                              filter_counts=[num_of_filters],
                              strides=[(1, 1)],
                              pool_dims=[2])
        # Dense layers
        net = tf.keras.layers.Flatten()(net)
        net = self.dense_block(node=node, input_net=net, dims=self.hiddenLayers, use_dropout=True)

        # Output
        self.nodeOutputsDict[node.index]["F"] = net
        self.nodeOutputsDict[node.index]["H"] = None
