import tensorflow as tf
from simple_tf.global_params import GlobalConstants
from simple_tf.global_params import GradientType


def root_func(node, network, variables=None):
    # Parameters
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        conv_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv_weight",
                                                                                                 node=node))
    else:
        conv_weights = tf.Variable(
            tf.constant(0.1, shape=[5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1],
                        dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_weight", node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, network.treeDegree],
                                stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = tf.Variable(
            tf.constant(value=0.1, shape=[GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, network.treeDegree]),
            name=network.get_variable_name(name="hyperplane_weights", node=node))

    # print_op = tf.Print(input_=network.dataTensor, data=[network.dataTensor], message="Print at Node:{0}".format(node.index))
    # node.evalDict[network.get_variable_name(name="Print", node=node)] = print_op
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[network.treeDegree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.variablesSet = {conv_weights, conv_biases, hyperplane_weights, hyperplane_biases}
    # Operations
    network.mask_input_nodes(node=node)
    # F
    conv = tf.nn.conv2d(network.dataTensor, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    flat_data = tf.contrib.layers.flatten(network.dataTensor)
    node.hOpsList.extend([flat_data])
    # Decisions
    node.activationsDict[node.index] = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
    network.apply_decision(node=node)


def l1_func(node, network, variables=None):
    # Parameters
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        conv_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2], stddev=0.1,
                                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_weight", node=node))
    else:
        conv_weights = tf.Variable(
            tf.constant(0.1, shape=[5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2],
                        dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_weight", node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        hyperplane_weights = tf.Variable(
            tf.truncated_normal(
                [GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE + network.treeDegree, network.treeDegree],
                stddev=0.1,
                seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE + network.treeDegree,
                                    network.treeDegree], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[network.treeDegree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.variablesSet = {conv_weights, conv_biases, hyperplane_weights, hyperplane_biases}
    # Operations
    parent_F, parent_H = network.mask_input_nodes(node=node)
    # F
    conv = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    node.hOpsList.extend([parent_H])
    # Decisions
    concat_list = [parent_H]
    concat_list.extend(node.activationsDict.values())
    h_concat = tf.concat(values=concat_list, axis=1)
    node.activationsDict[node.index] = tf.matmul(h_concat, hyperplane_weights) + hyperplane_biases
    network.apply_decision(node=node)


def leaf_func(node, network, variables=None):
    # Parameters
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        fc_weights_1 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
             GlobalConstants.NO_HIDDEN],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_1", node=node))
        fc_weights_2 = tf.Variable(
            tf.truncated_normal([GlobalConstants.NO_HIDDEN + 2 * network.treeDegree, GlobalConstants.NUM_LABELS],
                                stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_2", node=node))
    else:
        fc_weights_1 = tf.Variable(
            tf.constant(0.1, shape=[
                GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
                GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_1", node=node))
        fc_weights_2 = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN + 2 * network.treeDegree, GlobalConstants.NUM_LABELS],
                        dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_2", node=node))
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_1", node=node))
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_2", node=node))
    node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2}
    # Operations
    # Mask inputs
    parent_F, parent_H = network.mask_input_nodes(node=node)
    # Loss
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # Concatenate activations from ancestors
    concat_list = [hidden_layer]
    concat_list.extend(node.activationsDict.values())
    hidden_layer_concat = tf.concat(values=concat_list, axis=1)
    logits = tf.matmul(hidden_layer_concat, fc_weights_2) + fc_biases_2
    # Apply loss
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                               logits=logits)
    parallel_dnn_updates = {GradientType.parallel_dnns_unbiased, GradientType.parallel_dnns_biased}
    mixture_of_expert_updates = {GradientType.mixture_of_experts_biased, GradientType.mixture_of_experts_unbiased}
    if GlobalConstants.GRADIENT_TYPE in parallel_dnn_updates:
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
    elif GlobalConstants.GRADIENT_TYPE in mixture_of_expert_updates:
        pre_loss = tf.reduce_sum(cross_entropy_loss_tensor)
        loss = (1.0 / float(GlobalConstants.BATCH_SIZE)) * pre_loss
    else:
        raise NotImplementedError()
    node.fOpsList.extend([flattened, hidden_layer, hidden_layer_concat, logits, cross_entropy_loss_tensor, pre_loss,
                          loss])
    node.lossList.append(loss)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor
