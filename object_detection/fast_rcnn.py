import tensorflow as tf
import numpy as np

from algorithms.resnet.resnet_generator import ResnetGenerator
from object_detection.constants import Constants


class FastRcnn:
    def __init__(self):
        self.imageInputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')

    def build_resnet_backbone(self):
        # ResNet Parameters
        num_of_units_per_block = Constants.NUM_OF_RESIDUAL_UNITS
        num_of_feature_maps_per_block = Constants.NUM_OF_FEATURES_PER_BLOCK
        first_conv_filter_size = Constants.FIRST_CONV_FILTER_SIZE
        relu_leakiness = Constants.RELU_LEAKINESS
        stride_list = Constants.FILTER_STRIDES
        active_before_residuals = Constants.ACTIVATE_BEFORE_RESIDUALS


        # strides = GlobalConstants.RESNET_HYPERPARAMS.strides
        # activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
        # filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
        # num_of_units_per_block = Constants.num_residual_units
        # relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
        # first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size

        # Input layer
        x = ResnetGenerator.get_input(input=self.imageInputs, out_filters=num_of_feature_maps_per_block[0],
                                      first_conv_filter_size=first_conv_filter_size, node=node)




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