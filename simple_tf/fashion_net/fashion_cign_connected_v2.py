import tensorflow as tf

from auxillary.parameters import DiscreteParameter, DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants


def apply_router_transformation(net, network, node, decision_feature_size, node_degree):
    h_net = net
    net_shape = h_net.get_shape().as_list()
    # Global Average Pooling
    h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    feature_size = h_net.get_shape().as_list()[-1]
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, decision_feature_size],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights", node=node))
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias", node=node))
    node.variablesSet.add(fc_h_weights)
    node.variablesSet.add(fc_h_bias)
    h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
    h_net = tf.nn.relu(h_net)
    h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
    ig_feature = h_net
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
    if GlobalConstants.USE_UNIFIED_BATCH_NORM:
        network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature,
                                                       hyperplane_weights=hyperplane_weights,
                                                       hyperplane_biases=hyperplane_biases)
    else:
        network.apply_decision(node=node, branching_feature=ig_feature, hyperplane_weights=hyperplane_weights,
                               hyperplane_biases=hyperplane_biases)


def root_func(node, network, variables=None):
    # Parameters
    node_degree = network.degreeList[node.depth]
    # Convolution 1
    # OK
    conv1_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_1_SIZE,
                             GlobalConstants.NUM_CHANNELS, GlobalConstants.FASHION_F_NUM_FILTERS_1], stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE), name=network.get_variable_name(name="conv1_weight",
                                                                                             node=node))
    # OK
    conv1_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv1_bias",
                                       node=node))

    node.variablesSet = {conv1_weights, conv1_biases}
    # ***************** F: Convolution Layers *****************
    # First Conv Layer
    network.mask_input_nodes(node=node)
    net = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([net])
    # ***************** F: Convolution Layers *****************

    # ***************** H: Connected to F *****************
    apply_router_transformation(net=net, network=network, node=node,
                                decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1,
                                node_degree=node_degree)
    # ***************** H: Connected to F *****************


def l1_func(node, network, variables=None):
    # Parameters
    node_degree = network.degreeList[node.depth]
    # Convolution 2
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                             GlobalConstants.FASHION_F_NUM_FILTERS_1, GlobalConstants.FASHION_F_NUM_FILTERS_2],
                            stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv2_weight", node=node))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv2_bias",
                                       node=node))
    node.variablesSet = {conv_weights, conv_biases}
    # ***************** F: Convolution Layer *****************
    parent_F, parent_H = network.mask_input_nodes(node=node)
    net = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([net])
    # ***************** F: Convolution Layer *****************

    # ***************** H: Connected to F *****************
    apply_router_transformation(net=net, network=network, node=node,
                                decision_feature_size=GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2,
                                node_degree=node_degree)
    # ***************** H: Connected to F *****************


def leaf_func(node, network, variables=None):
    total_prev_degrees = sum(network.degreeList[0:node.depth])
    softmax_input_dim = GlobalConstants.FASHION_F_FC_2
    conv_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_3_SIZE, GlobalConstants.FASHION_FILTERS_3_SIZE,
                             GlobalConstants.FASHION_F_NUM_FILTERS_2, GlobalConstants.FASHION_F_NUM_FILTERS_3],
                            stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv3_weight", node=node))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_3], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv3_bias",
                                       node=node))
    node.variablesSet = {conv_weights, conv_biases}
    # ***************** F: Convolution Layer *****************
    # Conv Layer
    parent_F, parent_H = network.mask_input_nodes(node=node)
    net = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC Layers
    flattened = tf.contrib.layers.flatten(parent_F)
    flattened_F_feature_size = flattened.get_shape().as_list()[-1]
    # Parameters
    # OK
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [flattened_F_feature_size,
         GlobalConstants.FASHION_F_FC_1],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_weights_1", node=node))
    # OK
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_1], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_1", node=node))
    # OK
    fc_weights_2 = tf.Variable(tf.truncated_normal(
        [GlobalConstants.FASHION_F_FC_1,
         GlobalConstants.FASHION_F_FC_2],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_weights_2", node=node))
    # OK
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_F_FC_2], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_biases_2", node=node))
    # OK
    fc_softmax_weights = tf.Variable(
        tf.truncated_normal([softmax_input_dim, GlobalConstants.NUM_LABELS],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_softmax_weights", node=node))
    # OK
    fc_softmax_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_softmax_biases", node=node))
    node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2, fc_softmax_weights, fc_softmax_biases}
    # OPS
    hidden_layer_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    dropped_layer_1 = tf.nn.dropout(hidden_layer_1, network.classificationDropoutKeepProb)
    hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
    dropped_layer_2 = tf.nn.dropout(hidden_layer_2, network.classificationDropoutKeepProb)
    final_feature, logits = network.apply_loss(node=node, final_feature=dropped_layer_2,
                                               softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
    node.fOpsList.extend([net])
    # Evaluation
    node.evalDict[network.get_variable_name(name="final_eval_feature", node=node)] = final_feature
    node.evalDict[network.get_variable_name(name="logits", node=node)] = logits
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="fc_softmax_weights", node=node)] = fc_softmax_weights
    node.evalDict[network.get_variable_name(name="fc_softmax_biases", node=node)] = fc_softmax_biases
    # ***************** F: Convolution Layer *****************


def residue_network_func(network):
    all_residue_features, input_labels, input_indices = network.prepare_residue_input_tensors()
    network.residueInputTensor = all_residue_features  # tf.stop_gradient(all_residue_features)
    # Residue Network Parameters
    variable_list = []
    curr_layer = network.residueInputTensor
    for layer_index in range(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT):
        input_dim = curr_layer.get_shape().as_list()[-1]
        fc_residue_weights = tf.Variable(
            tf.truncated_normal([input_dim, GlobalConstants.FASHION_F_RESIDUE], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE), name="fc_residue_weights_{0}".format(layer_index))
        fc_residue_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_F_RESIDUE],
                                                  dtype=GlobalConstants.DATA_TYPE),
                                      name="fc_residue_bias_{0}".format(layer_index))
        variable_list.extend([fc_residue_weights, fc_residue_bias])
        curr_layer = tf.nn.relu(tf.matmul(curr_layer, fc_residue_weights) + fc_residue_bias)
        if GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT:
            curr_layer = tf.nn.dropout(curr_layer, keep_prob=network.classificationDropoutKeepProb)
    # Loss layer
    input_dim = curr_layer.get_shape().as_list()[-1]
    fc_residue_softmax_weights = tf.Variable(
        tf.truncated_normal([input_dim, GlobalConstants.NUM_LABELS], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE), name="fc_residue_final_weights")
    fc_residue_softmax_bias = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
                                                      dtype=GlobalConstants.DATA_TYPE),
                                          name="fc_residue_final_bias")
    variable_list.extend([fc_residue_softmax_weights, fc_residue_softmax_bias])
    network.variableManager.add_variables_to_node(node=None, tf_variables=variable_list)
    curr_layer = tf.nn.dropout(curr_layer, keep_prob=network.classificationDropoutKeepProb)
    residue_logits = tf.matmul(curr_layer, fc_residue_softmax_weights) + fc_residue_softmax_bias
    cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels,
                                                                               logits=residue_logits)
    loss = tf.reduce_mean(cross_entropy_loss_tensor)
    network.evalDict["residue_probabilities"] = tf.nn.softmax(residue_logits)
    network.evalDict["residue_labels"] = input_labels
    network.evalDict["residue_indices"] = input_indices
    network.evalDict["residue_features"] = network.residueInputTensor
    return loss
    # return tf.constant(value=0.0)


def grad_func(network):
    # self.initOp = tf.global_variables_initializer()
    # sess.run(self.initOp)
    vars = network.variableManager.trainable_variables()
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
                if GlobalConstants.RESIDE_AFFECTS_WHOLE_NETWORK:
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
    network.decisionParamsDict = {}
    network.mainLossParamsDict = {}
    network.residueParamsDict = {}
    network.regularizationParamsDict = {}
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
    pass


def threshold_calculator_func(network):
    # Noise Coefficient
    network.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                           decay=0.0,
                                                           decay_period=1,
                                                           min_limit=0.0)
    # Decision Loss Coefficient
    # network.decisionLossCoefficientCalculator = DiscreteParameter(name="decision_loss_coefficient_calculator",
    #                                                               value=0.0,
    #                                                               schedule=[(12000, 1.0)])
    network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator", value=1.0)
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
