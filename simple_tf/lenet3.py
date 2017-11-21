import tensorflow as tf

from auxillary.parameters import DecayingParameter
from simple_tf import batch_norm
from simple_tf.global_params import GlobalConstants
from simple_tf.global_params import GradientType


def root_func(node, network, variables=None):
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
    node.variablesSet = {conv_weights, conv_biases}
    # Operations
    network.mask_input_nodes(node=node)
    # F
    conv = tf.nn.conv2d(network.dataTensor, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    if GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE:
        conv_h_weights = tf.Variable(
            tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_H_FILTERS_1], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv_decision_weight",
                                                                                                 node=node))
        conv_h_bias = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_H_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_decision_bias",
                                           node=node))
        node.variablesSet.add(conv_h_weights)
        node.variablesSet.add(conv_h_bias)
        conv_h = tf.nn.conv2d(network.dataTensor, conv_h_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu_h = tf.nn.relu(tf.nn.bias_add(conv_h, conv_h_bias))
        pool_h = tf.nn.max_pool(relu_h, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        flat_data = tf.contrib.layers.flatten(pool_h)
        feature_size = flat_data.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, GlobalConstants.NO_H_FC_UNITS_1],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_H_FC_UNITS_1], dtype=GlobalConstants.DATA_TYPE),
                                  name=network.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        raw_ig_feature = tf.matmul(flat_data, fc_h_weights) + fc_h_bias
        ig_feature = tf.nn.relu(raw_ig_feature)
        node.hOpsList.extend([conv_h, relu_h, pool_h])
    else:
        ig_feature = tf.contrib.layers.flatten(network.dataTensor)
        node.hOpsList.extend([ig_feature])
    ig_feature_size = ig_feature.get_shape().as_list()[-1]
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.variablesSet.add(hyperplane_weights)
    node.variablesSet.add(hyperplane_biases)
    # Decisions
    if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
        normed_data, assign_ops = batch_norm.batch_norm(x=ig_feature, iteration=network.iterationHolder,
                                                        is_decision_phase=network.isDecisionPhase,
                                                        is_training_phase=network.isTrain,
                                                        decay=GlobalConstants.BATCH_NORM_DECAY,
                                                        node=node, network=network)
        network.branchingBatchNormAssignOps.extend(assign_ops)
        normed_activations = tf.matmul(normed_data, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = normed_activations
    else:
        activations = tf.matmul(ig_feature, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
    network.apply_decision(node=node)


def l1_func(node, network, variables=None):
    if GlobalConstants.USE_CONCAT_TRICK:
        raise NotImplementedError()
    node_degree = network.degreeList[node.depth]
    total_prev_degrees = sum(network.degreeList[0:node.depth])
    # Parameters
    conv_weights = tf.Variable(
        tf.truncated_normal([5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2], stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_weight", node=node))
    conv_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="conv_bias",
                                                             node=node))
    node.variablesSet = {conv_weights, conv_biases}
    # Operations
    parent_F, parent_H = network.mask_input_nodes(node=node)
    # F
    conv = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv, relu, pool])
    # H
    if GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE:
        conv_h_weights = tf.Variable(
            tf.truncated_normal([3, 3, GlobalConstants.NO_H_FILTERS_1, GlobalConstants.NO_H_FILTERS_2], stddev=0.1,
                                seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv_decision_weight",
                                                                                                 node=node))
        conv_h_bias = tf.Variable(
            tf.constant(0.1, shape=[GlobalConstants.NO_H_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="conv_decision_bias",
                                           node=node))
        node.variablesSet.add(conv_h_weights)
        node.variablesSet.add(conv_h_bias)
        conv_h = tf.nn.conv2d(parent_H, conv_h_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu_h = tf.nn.relu(tf.nn.bias_add(conv_h, conv_h_bias))
        pool_h = tf.nn.max_pool(relu_h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        flat_data = tf.contrib.layers.flatten(pool_h)
        feature_size = flat_data.get_shape().as_list()[-1]
        fc_h_weights = tf.Variable(tf.truncated_normal(
            [feature_size, GlobalConstants.NO_H_FC_UNITS_2],
            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
            name=network.get_variable_name(name="fc_decision_weights", node=node))
        fc_h_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_H_FC_UNITS_2], dtype=GlobalConstants.DATA_TYPE),
                                  name=network.get_variable_name(name="fc_decision_bias", node=node))
        node.variablesSet.add(fc_h_weights)
        node.variablesSet.add(fc_h_bias)
        raw_ig_feature = tf.matmul(flat_data, fc_h_weights) + fc_h_bias
        ig_feature = tf.nn.relu(raw_ig_feature)
        node.hOpsList.extend([conv_h, relu_h, pool_h])
    else:
        node.hOpsList.extend([parent_H])
        ig_feature = parent_H
    ig_feature_size = ig_feature.get_shape().as_list()[-1]
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.variablesSet.add(hyperplane_weights)
    node.variablesSet.add(hyperplane_biases)
    # Decisions
    h_vector = ig_feature
    # Batch Norm
    if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
        normed_h_vector, assign_ops = batch_norm.batch_norm(x=h_vector, iteration=network.iterationHolder,
                                                            is_decision_phase=network.isDecisionPhase,
                                                            is_training_phase=network.isTrain,
                                                            decay=GlobalConstants.BATCH_NORM_DECAY,
                                                            node=node, network=network)
        network.branchingBatchNormAssignOps.extend(assign_ops)
        normed_activations = tf.matmul(normed_h_vector, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = normed_activations
    else:
        activations = tf.matmul(h_vector, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
    network.apply_decision(node=node)


def leaf_func(node, network, variables=None):
    total_prev_degrees = sum(network.degreeList[0:node.depth])
    # Parameters
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
         GlobalConstants.NO_HIDDEN],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_weights_1", node=node))
    fc_weights_2 = tf.Variable(
        tf.truncated_normal([GlobalConstants.NO_HIDDEN, GlobalConstants.NUM_LABELS],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
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
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # Loss
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
    if GlobalConstants.USE_INFO_GAIN_DECISION:
        for v in vars:
            if "hyperplane" in v.name or "gamma" in v.name or "beta" in v.name or "_decision_" in v.name:
                decision_vars_list.append(v)
                if "hyperplane" in v.name or "_decision_" in v.name:
                    regularization_vars_list.append(v)
            else:
                classification_vars_list.append(v)
                regularization_vars_list.append(v)
                # if not ("gamma" in v.name or "beta" in v.name):
    elif len(network.topologicalSortedNodes) == 1:
        for v in vars:
            classification_vars_list.append(v)
            if not ("gamma" in v.name or "beta" in v.name):
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
    if len(decision_vars_list) > 0:
        network.decisionGradients = tf.gradients(ys=network.decisionLoss, xs=decision_vars_list)
    else:
        network.decisionGradients = None
    network.regularizationGradients = tf.gradients(ys=network.regularizationLoss, xs=regularization_vars_list)


def tensorboard_func(network):
    # Info gain, branch probs
    for node in network.topologicalSortedNodes:
        if not node.isLeaf:
            info_gain_name = network.get_variable_name(name="info_gain", node=node)
            branch_prob_name = network.get_variable_name(name="p(n|x)", node=node)
            info_gain_output = network.evalDict[network.get_variable_name(name="info_gain", node=node)]
            branch_prob_output = network.evalDict[network.get_variable_name(name="p(n|x)", node=node)]
            network.decisionPathSummaries.append(tf.summary.scalar(info_gain_name, info_gain_output))
            network.decisionPathSummaries.append(tf.summary.histogram(branch_prob_name, branch_prob_output))
    # Hyperplane norms
    vars = tf.trainable_variables()
    for v in vars:
        if "hyperplane" in v.name:
            loss_name = "l2_loss_{0}".format(v.name)
            l2_loss_output = network.evalDict[loss_name]
            network.classificationPathSummaries.append(tf.summary.scalar(loss_name, l2_loss_output))


def threshold_calculator_func(network):
    for node in network.topologicalSortedNodes:
        if node.isLeaf:
            continue
        node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
        initial_value = 1.0 / float(node_degree)
        name = network.get_variable_name(name="prob_threshold_calculator", node=node)
        node.probThresholdCalculator = DecayingParameter(name=name, value=initial_value, decay=0.999, decay_period=1,
                                                         min_limit=0.0)
