import tensorflow as tf
from simple_tf.global_params import GlobalConstants


def baseline(node, network, variables=None):
    # Parameters
    if network.createNewVariables:
        conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv1_weight",
                                                                                                 node=node))
        conv1_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv1_bias",
                                           node=node))
        conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2], stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv2_weight", node=node))
        conv2_biases = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv2_bias",
                                           node=node))
        fc_weights_1 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
             GlobalConstants.NO_HIDDEN],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
                                   name=network.get_variable_name(name="fc_weights_1", node=node))
        fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
                                  name=network.get_variable_name(name="fc_biases_1", node=node))
        fc_weights_2 = tf.Variable(
            tf.truncated_normal([GlobalConstants.NO_HIDDEN + network.depth - 1, GlobalConstants.NUM_LABELS],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_2", node=node))
        fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
                                  name=network.get_variable_name(name="fc_biases_2", node=node))
        node.variablesList.extend(
            [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc_weights_1, fc_biases_1, fc_weights_2,
             fc_biases_2])
    else:
        node.variablesList = []
        node.variablesList.extend(variables)
        conv1_weights = node.variablesList[0]
        conv1_biases = node.variablesList[1]
        conv2_weights = node.variablesList[2]
        conv2_biases = node.variablesList[3]
        fc_weights_1 = node.variablesList[4]
        fc_biases_1 = node.variablesList[5]
        fc_weights_2 = node.variablesList[6]
        fc_biases_2 = node.variablesList[7]
    # Operations
    conv1 = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flattened = tf.contrib.layers.flatten(pool2)
    hidden_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    logits = tf.matmul(hidden_1, fc_weights_2) + fc_biases_2
    # Loss
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=network.labelTensor,
                                                                               logits=logits)
    loss = tf.reduce_mean(cross_entropy_loss_tensor)
    node.fOpsList.extend([conv1, relu1, pool1, conv2, relu2, pool2, flattened, hidden_1, logits,
                          cross_entropy_loss_tensor, loss])
    node.lossList.append(loss)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = network.labelTensor
