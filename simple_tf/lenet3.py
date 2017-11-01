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
        conv_weights = network.get_fixed_variable(name="conv_weight", node=node)
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, network.treeDegree],
                                stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = network.get_fixed_variable(name="hyperplane_weights", node=node)
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
        conv_weights = network.get_fixed_variable(name="conv_weight", node=node)
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        if GlobalConstants.USE_CONCAT_TRICK:
            hyperplane_weights = tf.Variable(
                tf.truncated_normal(
                    [GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE + network.treeDegree, network.treeDegree],
                    stddev=0.1,
                    seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="hyperplane_weights", node=node))
        else:
            hyperplane_weights = tf.Variable(
                tf.truncated_normal(
                    [GlobalConstants.IMAGE_SIZE * GlobalConstants.IMAGE_SIZE, network.treeDegree],
                    stddev=0.1,
                    seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="hyperplane_weights", node=node))
    else:
        hyperplane_weights = network.get_fixed_variable(name="hyperplane_weights", node=node)
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
    if GlobalConstants.USE_CONCAT_TRICK:
        concat_list = [parent_H]
        concat_list.extend(node.activationsDict.values())
        h_concat = tf.concat(values=concat_list, axis=1)
        node.activationsDict[node.index] = tf.matmul(h_concat, hyperplane_weights) + hyperplane_biases
    else:
        node.activationsDict[node.index] = tf.matmul(parent_H, hyperplane_weights) + hyperplane_biases
    network.apply_decision(node=node)


def leaf_func(node, network, variables=None):
    # Parameters
    if GlobalConstants.USE_RANDOM_PARAMETERS:
        fc_weights_1 = tf.Variable(tf.truncated_normal(
            [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
             GlobalConstants.NO_HIDDEN],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_weights_1", node=node))
        if GlobalConstants.USE_CONCAT_TRICK:
            fc_weights_2 = tf.Variable(
                tf.truncated_normal([GlobalConstants.NO_HIDDEN + 2 * network.treeDegree, GlobalConstants.NUM_LABELS],
                                    stddev=0.1,
                                    seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="fc_weights_2", node=node))
        else:
            fc_weights_2 = tf.Variable(
                tf.truncated_normal([GlobalConstants.NO_HIDDEN, GlobalConstants.NUM_LABELS],
                                    stddev=0.1,
                                    seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=network.get_variable_name(name="fc_weights_2", node=node))

    else:
        fc_weights_1 = network.get_fixed_variable(name="fc_weights_1", node=node)
        fc_weights_2 = network.get_fixed_variable(name="fc_weights_2", node=node)
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_1", node=node))
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_2", node=node))
    node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2}
    # Operations
    # Mask inputs
    parent_F, parent_H = network.mask_input_nodes(node=node)
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # Loss
    if GlobalConstants.USE_CONCAT_TRICK:
        # Concatenate activations from ancestors
        concat_list = [hidden_layer]
        concat_list.extend(node.activationsDict.values())
        hidden_layer_concat = tf.concat(values=concat_list, axis=1)
        logits = tf.matmul(hidden_layer_concat, fc_weights_2) + fc_biases_2
        node.fOpsList.extend([flattened, hidden_layer, hidden_layer_concat, logits])
    else:
        logits = tf.matmul(hidden_layer, fc_weights_2) + fc_biases_2
        node.fOpsList.extend([flattened, hidden_layer, logits])
    # Apply loss
    network.apply_loss(node=node, logits=logits)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor


def grad_func(network):
    # self.initOp = tf.global_variables_initializer()
    # sess.run(self.initOp)
    vars = tf.trainable_variables()
    decision_vars_list = []
    classification_vars_list = []
    regularization_vars_list = []
    if GlobalConstants.USE_CONCAT_TRICK:
        for v in vars:
            if "hyperplane" in v.name:
                decision_vars_list.append(v)
            classification_vars_list.append(v)
            regularization_vars_list.append(v)
    elif GlobalConstants.USE_INFO_GAIN_DECISION:
        for v in vars:
            if "hyperplane" in v.name:
                decision_vars_list.append(v)
            else:
                classification_vars_list.append(v)
            regularization_vars_list.append(v)
    else:
        raise NotImplementedError()
    for i in range(len(decision_vars_list)):
        network.decisionParamsDict[decision_vars_list[i]] = i
    for i in range(len(classification_vars_list)):
        network.mainLossParamsDict[classification_vars_list[i]] = i
    for i in range(len(regularization_vars_list)):
        network.regularizationParamsDict[regularization_vars_list[i]] = i
    # for i in range(len(vars)):
    #     network.regularizationParamsDict[vars[i]] = i
    network.classificationGradients = tf.gradients(ys=network.mainLoss, xs=classification_vars_list)
    network.decisionGradients = tf.gradients(ys=network.decisionLoss, xs=decision_vars_list)
    network.regularizationGradients = tf.gradients(ys=network.regularizationLoss, xs=regularization_vars_list)
