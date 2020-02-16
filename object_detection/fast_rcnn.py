import tensorflow as tf
import numpy as np

from object_detection.constants import Constants
from object_detection.residual_network_generator import ResidualNetworkGenerator


class FastRcnn:
    def __init__(self, backbone_type):
        self.imageInputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
        self.isTrain = tf.placeholder(name="is_train_flag", dtype=tf.int64)
        self.backboneType = backbone_type
        self.backboneNetworkOutput = None

    def build_network(self):
        # Build the backbone
        if self.backboneType == "Resnet":
            self.backboneNetworkOutput = self.build_resnet_backbone()
        else:
            raise NotImplementedError()

    def build_resnet_backbone(self):
        # ResNet Parameters
        num_of_units_per_block = Constants.NUM_OF_RESIDUAL_UNITS
        num_of_feature_maps_per_block = Constants.NUM_OF_FEATURES_PER_BLOCK
        first_conv_filter_size = Constants.FIRST_CONV_FILTER_SIZE
        relu_leakiness = Constants.RELU_LEAKINESS
        stride_list = Constants.FILTER_STRIDES
        active_before_residuals = Constants.ACTIVATE_BEFORE_RESIDUALS

        assert len(num_of_feature_maps_per_block) == len(stride_list) + 1 and \
               len(num_of_feature_maps_per_block) == len(active_before_residuals) + 1
        # Input layer
        x = ResidualNetworkGenerator.get_input(input=self.imageInputs, out_filters=num_of_feature_maps_per_block[0],
                                               first_conv_filter_size=first_conv_filter_size)
        # Loop over blocks, the resnet trunk
        for block_id in range(len(num_of_feature_maps_per_block) - 1):
            with tf.variable_scope("block_{0}_0".format(block_id)):
                x = ResidualNetworkGenerator.bottleneck_residual(
                    x=x,
                    in_filter=num_of_feature_maps_per_block[block_id],
                    out_filter=num_of_feature_maps_per_block[block_id + 1],
                    stride=ResidualNetworkGenerator.stride_arr(stride_list[block_id]),
                    activate_before_residual=active_before_residuals[block_id],
                    relu_leakiness=relu_leakiness,
                    is_train=self.isTrain,
                    bn_momentum=Constants.BATCH_NORM_DECAY)
            for i in range(num_of_units_per_block - 1):
                with tf.variable_scope("block_{0}_{1}".format(block_id, i + 1)):
                    x = ResidualNetworkGenerator.bottleneck_residual(
                        x=x,
                        in_filter=num_of_feature_maps_per_block[block_id + 1],
                        out_filter=num_of_feature_maps_per_block[block_id + 1],
                        stride=ResidualNetworkGenerator.stride_arr(1),
                        activate_before_residual=False,
                        relu_leakiness=relu_leakiness,
                        is_train=self.isTrain,
                        bn_momentum=Constants.BATCH_NORM_DECAY)
        return x


# net = imageInputs
# in_filters = imageInputs.get_shape().as_list()[-1]
# out_filters = 32
# pooled_height = 7
# pooled_width = 7
#
# W = tf.get_variable("W", [3, 3, in_filters, out_filters], trainable=True)
# b = tf.get_variable("b", [out_filters], trainable=True)
# net = tf.nn.conv2d(net, W, strides = [1, 2, 2, 1], padding='SAME')
# net = tf.nn.bias_add(net, b)
# net = tf.nn.relu(net)
#
# X = np.random.uniform(low=0.0, high=1.0, size=(3, 2500, 640, 3))
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# results = sess.run([net], feed_dict={imageInputs: X})
#
# print("X")
