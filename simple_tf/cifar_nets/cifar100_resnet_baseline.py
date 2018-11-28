import tensorflow as tf
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator


def baseline(node, network, variables=None):
    network.mask_input_nodes(node=node)
    # Input layer
    resnet_first_layer = ResnetGenerator.get_input(input=network.dataTensor, node=node, conv_filter_size=3)