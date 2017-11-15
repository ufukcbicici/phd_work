import tensorflow as tf

from simple_tf import batch_norm
from simple_tf.global_params import GlobalConstants
from simple_tf.global_params import GradientType


def root_func(node, network):
    # Parameters
    node_degree = network.degreeList[node.depth]
    conv_weights = tf.Variable(
        tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1], stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv_weight",
                                                                                             node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    # Hyperplanes
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, node_degree],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="hyperplane_weights", node=node))

    hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    # Operations
    # F
    conv = tf.nn.conv2d(network.dataTensor, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    flat_data = tf.contrib.layers.flatten(network.dataTensor)
    node.hOpsList.extend([flat_data])
    # Calculate branching probabilities
    activations = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    node.activationsDict[node.index] = activations
    network.apply_decision(node=node)


    # if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
    #     normed_data, assign_ops = batch_norm.batch_norm(x=flat_data, iteration=network.iterationHolder,
    #                                                     is_decision_phase=network.isDecisionPhase,
    #                                                     is_training_phase=network.isTrain,
    #                                                     decay=GlobalConstants.BATCH_NORM_DECAY,
    #                                                     node=node, network=network)
    #     network.branchingBatchNormAssignOps.extend(assign_ops)
    #     normed_activations = tf.matmul(normed_data, hyperplane_weights) + hyperplane_biases
    #     node.activationsDict[node.index] = normed_activations
    # else:
    #     activations = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    #     node.activationsDict[node.index] = activations
    # network.apply_decision(node=node)