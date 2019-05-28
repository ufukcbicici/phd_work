import tensorflow as tf
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DecayingParameter, FixedParameter
from simple_tf.global_params import GlobalConstants
from simple_tf.resnet_experiments.resnet_generator import ResnetGenerator

strides = GlobalConstants.RESNET_HYPERPARAMS.strides
activate_before_residual = GlobalConstants.RESNET_HYPERPARAMS.activate_before_residual
filters = GlobalConstants.RESNET_HYPERPARAMS.num_of_features_per_block
num_of_units_per_block = GlobalConstants.RESNET_HYPERPARAMS.num_residual_units
relu_leakiness = GlobalConstants.RESNET_HYPERPARAMS.relu_leakiness
first_conv_filter_size = GlobalConstants.RESNET_HYPERPARAMS.first_conv_filter_size


def apply_router_transformation(net, node, network, decision_feature_size):
    h_net = net
    net_shape = h_net.get_shape().as_list()
    # Global Average Pooling
    h_net = tf.nn.avg_pool(h_net, ksize=[1, net_shape[1], net_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    net_shape = h_net.get_shape().as_list()
    h_net = tf.reshape(h_net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    feature_size = h_net.get_shape().as_list()[-1]
    # MultiGPU OK
    fc_h_weights = UtilityFuncs.create_variable(
        name=network.get_variable_name(name="fc_decision_weights", node=node),
        shape=[feature_size, decision_feature_size],
        dtype=GlobalConstants.DATA_TYPE,
        initializer=tf.truncated_normal(
        [feature_size, decision_feature_size], stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    # MultiGPU OK
    fc_h_bias = UtilityFuncs.create_variable(
        name=network.get_variable_name(name="fc_decision_bias", node=node),
        shape=[decision_feature_size],
        dtype=GlobalConstants.DATA_TYPE,
        initializer=tf.constant(0.1, shape=[decision_feature_size], dtype=GlobalConstants.DATA_TYPE))
    h_net = tf.matmul(h_net, fc_h_weights) + fc_h_bias
    h_net = tf.nn.relu(h_net)
    h_net = tf.nn.dropout(h_net, keep_prob=network.decisionDropoutKeepProb)
    ig_feature = h_net
    node.hOpsList.extend([ig_feature])
    # Decisions
    if GlobalConstants.USE_UNIFIED_BATCH_NORM:
        network.apply_decision_with_unified_batch_norm(node=node, branching_feature=ig_feature)
    else:
        network.apply_decision(node=node, branching_feature=ig_feature)


# MultiGPU OK
def root_func(node, network):
    network.mask_input_nodes(node=node)
    # Input layer
    # MultiGPU OK
    x = ResnetGenerator.get_input(input=network.dataTensor, out_filters=filters[0],
                                  first_conv_filter_size=first_conv_filter_size)
    # Block 1
    # MultiGPU OK
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_1_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[0], out_filter=filters[1],
                                                stride=ResnetGenerator.stride_arr(strides[0]),
                                                activate_before_residual=activate_before_residual[0],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # MultiGPU OK
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_1_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[1],
                                                    out_filter=filters[1],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    node.fOpsList.extend([x])
    # MultiGPU OK
    # ***************** H: Connected to F *****************
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="decision_variables", node=node)):
        apply_router_transformation(net=x, network=network, node=node,
                                    decision_feature_size=GlobalConstants.RESNET_DECISION_DIMENSION)
    # ***************** H: Connected to F *****************


# MultiGPU OK
def l1_func(node, network):
    # MultiGPU OK
    parent_F, parent_H = network.mask_input_nodes(node=node)
    x = parent_F
    # MultiGPU OK
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_2_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[1], out_filter=filters[2],
                                                stride=ResnetGenerator.stride_arr(strides[1]),
                                                activate_before_residual=activate_before_residual[1],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # MultiGPU OK
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_2_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[2],
                                                    out_filter=filters[2],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    node.fOpsList.extend([x])
    # MultiGPU OK
    # ***************** H: Connected to F *****************
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="decision_variables", node=node)):
        apply_router_transformation(net=x, network=network, node=node,
                                    decision_feature_size=GlobalConstants.RESNET_DECISION_DIMENSION)
    # ***************** H: Connected to F *****************


# MultiGPU OK
def leaf_func(node, network):
    # MultiGPU OK
    parent_F, parent_H = network.mask_input_nodes(node=node)
    x = parent_F
    # MultiGPU OK
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_3_0", node=node)):
        x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[2], out_filter=filters[3],
                                                stride=ResnetGenerator.stride_arr(strides[2]),
                                                activate_before_residual=activate_before_residual[2],
                                                relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # MultiGPU OK
    for i in range(num_of_units_per_block-1):
        with tf.variable_scope(UtilityFuncs.get_variable_name(name="block_3_{0}".format(i + 1), node=node)):
            x = ResnetGenerator.bottleneck_residual(x=x, in_filter=filters[3],
                                                    out_filter=filters[3],
                                                    stride=ResnetGenerator.stride_arr(1),
                                                    activate_before_residual=False,
                                                    relu_leakiness=relu_leakiness, is_train=network.isTrain,
                                                    bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # Logit Layers
    with tf.variable_scope(UtilityFuncs.get_variable_name(name="unit_last", node=node)):
        x = ResnetGenerator.get_output(x=x, is_train=network.isTrain, leakiness=relu_leakiness,
                                       bn_momentum=GlobalConstants.BATCH_NORM_DECAY)
    # assert len(net_shape) == 4
    # x = tf.reshape(x, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    output = x
    out_dim = network.labelCount
    # MultiGPU OK
    weight = UtilityFuncs.create_variable(
        name=UtilityFuncs.get_variable_name(name="fc_softmax_weights", node=node),
        shape=[output.get_shape()[1], out_dim],
        dtype=GlobalConstants.DATA_TYPE,
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    # MultiGPU OK
    bias = UtilityFuncs.create_variable(
        name=UtilityFuncs.get_variable_name(name="fc_softmax_biases", node=node),
        shape=[out_dim],
        dtype=GlobalConstants.DATA_TYPE,
        initializer=tf.constant_initializer())
    # Loss
    # MultiGPU OK
    final_feature, logits = network.apply_loss(node=node, final_feature=output,
                                               softmax_weights=weight, softmax_biases=bias)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor


def residue_network_func(network):
    return tf.constant(0.0)


def grad_func(network):
    pass


def tensorboard_func(network):
    pass


def hyperparameter_func(network, **kwargs):
    GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
    GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]

    if not network.isBaseline:




        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                       value=kwargs["decision_keep_probability"])







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
            # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
            #                                                  decay_period=70000,
            #                                                  min_limit=0.4)
            node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
            # Softmax Decay
            decay_name = network.get_variable_name(name="softmax_decay", node=node)
            node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                            value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                            decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                            decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                            min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)


def hyperparameter_func_sampling(network, hyperparameter_func):
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
        # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
        #                                                  decay_period=70000,
        #                                                  min_limit=0.4)
        node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
        # Softmax Decay
        decay_name = network.get_variable_name(name="softmax_decay", node=node)
        node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                        value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                        decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                        decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                        min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)
