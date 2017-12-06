import tensorflow as tf
import numpy as np

from auxillary.parameters import DecayingParameter, DiscreteParameter, FixedParameter
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
    pool_h = tf.nn.max_pool(relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    flat_pool = tf.contrib.layers.flatten(pool_h)
    feature_size = flat_pool.get_shape().as_list()[-1]
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, GlobalConstants.NO_H_FROM_F_UNITS_1],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights", node=node))
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.NO_H_FC_UNITS_1], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias", node=node))
    node.variablesSet.add(fc_h_weights)
    node.variablesSet.add(fc_h_bias)
    raw_ig_feature = tf.matmul(flat_pool, fc_h_weights) + fc_h_bias
    # ***************** Dropout *****************
    relu_ig_feature = tf.nn.relu(raw_ig_feature)
    drooped_ig_feature = tf.nn.dropout(relu_ig_feature, keep_prob=network.decisionDropoutKeepProb)
    ig_feature = drooped_ig_feature
    # ***************** Dropout *****************
    node.hOpsList.extend([pool_h, flat_pool, raw_ig_feature, relu_ig_feature, drooped_ig_feature, ig_feature])
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
    network.apply_decision(node=node, branching_feature=ig_feature, hyperplane_weights=hyperplane_weights,
                           hyperplane_biases=hyperplane_biases)


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
    pool_h = tf.nn.max_pool(relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    flat_pool = tf.contrib.layers.flatten(pool_h)
    feature_size = flat_pool.get_shape().as_list()[-1]
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, GlobalConstants.NO_H_FROM_F_UNITS_2],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights", node=node))
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.NO_H_FC_UNITS_2], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias", node=node))
    node.variablesSet.add(fc_h_weights)
    node.variablesSet.add(fc_h_bias)
    raw_ig_feature = tf.matmul(flat_pool, fc_h_weights) + fc_h_bias
    # ig_feature = tf.nn.relu(raw_ig_feature)
    # ***************** Dropout *****************
    relu_ig_feature = tf.nn.relu(raw_ig_feature)
    drooped_ig_feature = tf.nn.dropout(relu_ig_feature, keep_prob=network.decisionDropoutKeepProb)
    ig_feature = drooped_ig_feature
    # ***************** Dropout *****************
    node.hOpsList.extend([pool_h, flat_pool, raw_ig_feature, relu_ig_feature, drooped_ig_feature, ig_feature])
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
    network.apply_decision(node=node, branching_feature=ig_feature, hyperplane_weights=hyperplane_weights,
                           hyperplane_biases=hyperplane_biases)


def leaf_func(node, network, variables=None):
    total_prev_degrees = sum(network.degreeList[0:node.depth])
    # Parameters
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
         GlobalConstants.NO_HIDDEN],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_weights_1", node=node))
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_1", node=node))
    softmax_input_dim = GlobalConstants.NO_HIDDEN
    if GlobalConstants.USE_DECISION_AUGMENTATION:
        softmax_input_dim += total_prev_degrees
    fc_softmax_weights = tf.Variable(
        tf.truncated_normal([softmax_input_dim, GlobalConstants.NUM_LABELS],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_softmax_weights", node=node))
    fc_softmax_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_softmax_biases", node=node))
    node.variablesSet = {fc_weights_1, fc_biases_1, fc_softmax_weights, fc_softmax_biases}
    # Operations
    # Mask inputs
    parent_F, parent_H = network.mask_input_nodes(node=node)
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    final_feature, logits = network.apply_loss(node=node, final_feature=hidden_layer,
                                               softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
    # Evaluation
    node.evalDict[network.get_variable_name(name="final_eval_feature", node=node)] = final_feature
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    # node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor


def residue_network_func(network):
    all_residue_features, input_labels, input_indices = network.prepare_residue_input_tensors()
    input_x = all_residue_features  # tf.stop_gradient(all_residue_features)
    input_dim = input_x.get_shape().as_list()[-1]
    # Residue Network Parameters
    fc_residue_weights_1 = tf.Variable(
        tf.truncated_normal([input_dim, 15], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_1")
    fc_residue_bias_1 = tf.Variable(tf.constant(0.1, shape=[15], dtype=GlobalConstants.DATA_TYPE),
                                    name="fc_residue_bias_1")
    fc_residue_weights_2 = tf.Variable(
        tf.truncated_normal([15, GlobalConstants.NUM_LABELS], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_2")
    fc_residue_bias_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
                                                dtype=GlobalConstants.DATA_TYPE), name="fc_residue_bias_2")
    # Reside Network Operations
    residue_hidden_layer = tf.nn.relu(tf.matmul(input_x, fc_residue_weights_1) + fc_residue_bias_1)
    residue_drop = tf.nn.dropout(residue_hidden_layer, keep_prob=network.classificationDropoutKeepProb)
    residue_logits = tf.matmul(residue_drop, fc_residue_weights_2) + fc_residue_bias_2
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels,
                                                                               logits=residue_logits)
    loss = tf.reduce_mean(cross_entropy_loss_tensor)
    network.evalDict["residue_probabilities"] = tf.nn.softmax(residue_logits)
    network.evalDict["residue_labels"] = input_labels
    network.evalDict["residue_indices"] = input_indices
    network.evalDict["residue_features"] = input_x
    return loss

    # conv_weights_1 = tf.Variable(
    #     tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, 20], stddev=0.1, seed=GlobalConstants.SEED,
    #                         dtype=GlobalConstants.DATA_TYPE), name="conv_residue_weights_1")
    # conv_biases_1 = tf.Variable(tf.constant(0.1, shape=[20], dtype=GlobalConstants.DATA_TYPE),
    #                           name="conv_residue_bias_1")
    # conv_weights_2 = tf.Variable(
    #     tf.truncated_normal([5, 5, 20, 50], stddev=0.1, seed=GlobalConstants.SEED,
    #                         dtype=GlobalConstants.DATA_TYPE), name="conv_residue_weights_2")
    # conv_biases_2 = tf.Variable(tf.constant(0.1, shape=[50], dtype=GlobalConstants.DATA_TYPE),
    #                           name="conv_residue_bias_2")
    #
    # conv1 = tf.nn.conv2d(network.dataTensor, conv_weights_1, strides=[1, 1, 1, 1], padding='SAME')
    # relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv_biases_1))
    # pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2 = tf.nn.conv2d(pool1, conv_weights_2, strides=[1, 1, 1, 1], padding='SAME')
    # relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv_biases_2))
    # pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # flattened = tf.contrib.layers.flatten(pool2)
    #
    # flat_dim = flattened.get_shape().as_list()[-1]
    # fc_residue_weights_1 = tf.Variable(
    #     tf.truncated_normal([flat_dim, 50], stddev=0.1, seed=GlobalConstants.SEED,
    #                         dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_1")
    # fc_residue_bias_1 = tf.Variable(tf.constant(0.1, shape=[50], dtype=GlobalConstants.DATA_TYPE),
    #                                 name="fc_residue_bias_1")
    # fc_residue_weights_2 = tf.Variable(
    #     tf.truncated_normal([50, GlobalConstants.NUM_LABELS], stddev=0.1, seed=GlobalConstants.SEED,
    #                         dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_2")
    # fc_residue_bias_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
    #                                             dtype=GlobalConstants.DATA_TYPE), name="fc_residue_bias_2")
    # hidden_layer = tf.nn.relu(tf.matmul(flattened, fc_residue_weights_1) + fc_residue_bias_1)
    # residue_logits = tf.matmul(hidden_layer, fc_residue_weights_2) + fc_residue_bias_2
    # cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=network.labelTensor,
    #                                                                            logits=residue_logits)
    # loss = tf.reduce_mean(cross_entropy_loss_tensor)
    # network.evalDict["residue_probabilities"] = tf.nn.softmax(residue_logits)
    # network.evalDict["residue_labels"] = network.labelTensor
    # network.evalDict["residue_indices"] = network.indicesTensor
    # network.evalDict["residue_features"] = network.dataTensor
    # return loss


def grad_func(network):
    # self.initOp = tf.global_variables_initializer()
    # sess.run(self.initOp)
    vars = tf.trainable_variables()
    decision_vars_list = []
    classification_vars_list = []
    residue_vars_list = []
    regularization_vars_list = []
    if GlobalConstants.USE_INFO_GAIN_DECISION:
        for v in vars:
            if "scale" in v.name or "shift" in v.name or "hyperplane" in v.name or \
                            "gamma" in v.name or "beta" in v.name or "_decision_" in v.name:
                decision_vars_list.append(v)
                if GlobalConstants.USE_DECISION_AUGMENTATION:
                    classification_vars_list.append(v)
                if "hyperplane" in v.name or "_decision_" in v.name:
                    regularization_vars_list.append(v)
            elif "_residue_" in v.name:
                residue_vars_list.append(v)
                regularization_vars_list.append(v)
            else:
                classification_vars_list.append(v)
                if "_softmax_" not in v.name:
                    residue_vars_list.append(v)
                if "conv" in v.name:
                    decision_vars_list.append(v)
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
    for i in range(len(residue_vars_list)):
        network.residueParamsDict[residue_vars_list[i]] = i
    for i in range(len(regularization_vars_list)):
        network.regularizationParamsDict[regularization_vars_list[i]] = i
    # for i in range(len(vars)):
    #     network.regularizationParamsDict[vars[i]] = i
    network.classificationGradients = tf.gradients(ys=network.mainLoss, xs=classification_vars_list)
    if len(decision_vars_list) > 0:
        network.decisionGradients = tf.gradients(ys=network.decisionLoss, xs=decision_vars_list)
    else:
        network.decisionGradients = None
    network.residueGradients = tf.gradients(ys=network.residueLoss, xs=residue_vars_list)
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
    # Decision Dropout Schedule
    # network.decisionDropoutKeepProbCalculator = DiscreteParameter(name="decision_dropout_calculator",
    #                                                               schedule=GlobalConstants.DROPOUT_SCHEDULE,
    #                                                               value=GlobalConstants.DROPOUT_INITIAL_PROB)
    network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob", value=0.5)
    # Noise Coefficient
    network.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=1.0,
                                                           decay=0.99995,
                                                           decay_period=1,
                                                           min_limit=0.0)
    for node in network.topologicalSortedNodes:
        if node.isLeaf:
            continue
        # Probability Threshold
        node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
        initial_value = 1.0 / float(node_degree)
        threshold_name = network.get_variable_name(name="prob_threshold_calculator", node=node)
        node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
                                                         decay_period=12000,
                                                         min_limit=0.4)
        # Softmax Decay
        decay_name = network.get_variable_name(name="softmax_decay", node=node)
        node.softmaxDecayCalculator = DecayingParameter(name=decay_name, value=GlobalConstants.SOFTMAX_DECAY_INITIAL,
                                                        decay=GlobalConstants.SOFTMAX_DECAY_COEFFICIENT,
                                                        decay_period=GlobalConstants.SOFTMAX_DECAY_PERIOD,
                                                        min_limit=GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
