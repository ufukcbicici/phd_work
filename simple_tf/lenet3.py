import tensorflow as tf
from simple_tf.global_params import GlobalConstants


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
    node.variablesList.extend([conv_weights, conv_biases, hyperplane_weights, hyperplane_biases])
    # Operations
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
    node.variablesList.extend([conv_weights, conv_biases, hyperplane_weights, hyperplane_biases])
    # Operations
    # Mask inputs
    parent_node = network.dagObject.parents(node=node)[0]
    # print_op = tf.Print(input_=parent_node.fOpsList[-1], data=[parent_node.fOpsList[-1]], message="Print at Node:{0}".format(node.index))
    # node.evalDict[network.get_variable_name(name="Print", node=node)] = print_op
    mask_tensor = parent_node.maskTensorsDict[node.index]
    if GlobalConstants.USE_CPU_MASKING:
        with tf.device("/cpu:0"):
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(network.labelTensor, mask_tensor)
    else:
        parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
        parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
        for k, v in parent_node.activationsDict.items():
            node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
        node.labelTensor = tf.boolean_mask(network.labelTensor, mask_tensor)
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
    node.variablesList.extend([fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2])
    # Operations
    # Mask inputs
    parent_node = network.dagObject.parents(node=node)[0]
    # print_op = tf.Print(input_=parent_node.fOpsList[-1], data=[parent_node.fOpsList[-1]], message="Print at Node:{0}".format(node.index))
    # node.evalDict[network.get_variable_name(name="Print", node=node)] = print_op
    mask_tensor = parent_node.maskTensorsDict[node.index]
    if GlobalConstants.USE_CPU_MASKING:
        with tf.device("/cpu:0"):
            parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
            # parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
            node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
    else:
        parent_F = tf.boolean_mask(parent_node.fOpsList[-1], mask_tensor)
        # parent_H = tf.boolean_mask(parent_node.hOpsList[-1], mask_tensor)
        for k, v in parent_node.activationsDict.items():
            node.activationsDict[k] = tf.boolean_mask(v, mask_tensor)
        node.labelTensor = tf.boolean_mask(parent_node.labelTensor, mask_tensor)
    # Loss
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # Concatenate activations from ancestors
    concat_list = [hidden_layer]
    concat_list.extend(node.activationsDict.values())
    hidden_layer_concat = tf.concat(values=concat_list, axis=1)
    logits = tf.matmul(hidden_layer_concat, fc_weights_2) + fc_biases_2
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                               logits=logits)
    pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
    loss = tf.where(tf.is_nan(pre_loss), 0., pre_loss)
    node.fOpsList.extend([flattened, hidden_layer, hidden_layer_concat, logits, cross_entropy_loss_tensor, pre_loss,
                          loss])
    node.lossList.append(loss)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor
    node.evalDict[network.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
