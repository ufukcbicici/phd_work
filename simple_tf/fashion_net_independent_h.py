import tensorflow as tf

from simple_tf.global_params import GlobalConstants


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
    # Convolution 2
    # OK
    conv2_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_FILTERS_2_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
                             GlobalConstants.FASHION_F_NUM_FILTERS_1, GlobalConstants.FASHION_F_NUM_FILTERS_2],
                            stddev=0.1,
                            seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv2_weight", node=node))
    # OK
    conv2_biases = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_F_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv2_bias",
                                       node=node))
    node.variablesSet = {conv1_weights, conv1_biases, conv2_weights, conv2_biases}
    # ***************** F: Convolution Layers *****************
    # First Conv Layer
    network.mask_input_nodes(node=node)
    conv1 = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Second Conv Layer
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv1, relu1, pool1, conv2, relu2, pool2])
    # ***************** F: Convolution Layers *****************

    # ***************** H: Convolution Layers *****************
    # OK
    conv_h_weights_1 = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_H_FILTERS_1_SIZE, GlobalConstants.FASHION_H_FILTERS_1_SIZE,
                             GlobalConstants.NUM_CHANNELS, GlobalConstants.FASHION_H_NUM_FILTERS_1], stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_weight_1",
                                       node=node))
    # OK
    conv_h_bias_1 = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_H_NUM_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_bias_1",
                                       node=node))
    # OK
    conv_h_weights_2 = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_H_FILTERS_2_SIZE, GlobalConstants.FASHION_H_FILTERS_2_SIZE,
                             GlobalConstants.FASHION_H_NUM_FILTERS_1, GlobalConstants.FASHION_H_NUM_FILTERS_2],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_weight_2",
                                       node=node))
    # OK
    conv_h_bias_2 = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_H_NUM_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_bias_2",
                                       node=node))
    # H Operations
    # OK
    conv_h_1 = tf.nn.conv2d(network.dataTensor, conv_h_weights_1, strides=[1, 1, 1, 1], padding='SAME')
    relu_h_1 = tf.nn.relu(tf.nn.bias_add(conv_h_1, conv_h_bias_1))
    pool_h_1 = tf.nn.max_pool(relu_h_1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    conv_h_2 = tf.nn.conv2d(pool_h_1, conv_h_weights_2, strides=[1, 1, 1, 1], padding='SAME')
    relu_h_2 = tf.nn.relu(tf.nn.bias_add(conv_h_2, conv_h_bias_2))
    pool_h_2 = tf.nn.max_pool(relu_h_2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    flat_data = tf.contrib.layers.flatten(pool_h_2)
    feature_size = flat_data.get_shape().as_list()[-1]
    # OK
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, GlobalConstants.FASHION_H_FC_1],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights", node=node))
    # OK
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_H_FC_1], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias", node=node))
    raw_ig_feature = tf.matmul(flat_data, fc_h_weights) + fc_h_bias
    ig_feature = tf.nn.relu(raw_ig_feature)
    ig_feature_size = ig_feature.get_shape().as_list()[-1]
    # OK
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="hyperplane_weights", node=node))
    hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.hOpsList.extend([conv_h_1, relu_h_1, pool_h_1, conv_h_2, relu_h_2, pool_h_2])
    node.variablesSet.add(conv_h_weights_1)
    node.variablesSet.add(conv_h_bias_1)
    node.variablesSet.add(conv_h_weights_2)
    node.variablesSet.add(conv_h_bias_2)
    node.variablesSet.add(fc_h_weights)
    node.variablesSet.add(fc_h_bias)
    node.variablesSet.add(hyperplane_weights)
    node.variablesSet.add(hyperplane_biases)
    # Decisions
    network.apply_decision(node=node, branching_feature=ig_feature, hyperplane_weights=hyperplane_weights,
                           hyperplane_biases=hyperplane_biases)
    # ***************** H: Convolution Layers *****************


def l1_func(node, network, variables=None):
    # Parameters
    node_degree = network.degreeList[node.depth]
    # Convolution
    # OK
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
    parent_F, parent_H = network.mask_input_nodes(node=node)
    conv3 = tf.nn.conv2d(parent_F, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv_biases))
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    node.fOpsList.extend([conv3, relu3, pool3])
    # ***************** F: Convolution Layer *****************

    # ***************** H: Convolution Layer *****************
    # OK
    conv_h_weights = tf.Variable(
        tf.truncated_normal([GlobalConstants.FASHION_H_FILTERS_3_SIZE, GlobalConstants.FASHION_H_FILTERS_3_SIZE,
                             GlobalConstants.FASHION_H_NUM_FILTERS_2, GlobalConstants.FASHION_H_NUM_FILTERS_3],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_weight_3",
                                       node=node))
    # OK
    conv_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_H_NUM_FILTERS_3], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="conv_decision_bias_3",
                                       node=node))
    # H Operations
    conv_h = tf.nn.conv2d(parent_H, conv_h_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu_h = tf.nn.relu(tf.nn.bias_add(conv_h, conv_h_bias))
    pool_h = tf.nn.max_pool(relu_h, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    flat_data = tf.contrib.layers.flatten(pool_h)
    feature_size = flat_data.get_shape().as_list()[-1]
    # OK
    fc_h_weights = tf.Variable(tf.truncated_normal(
        [feature_size, GlobalConstants.FASHION_H_FC_2],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_weights_2", node=node))
    # OK
    fc_h_bias = tf.Variable(
        tf.constant(0.1, shape=[GlobalConstants.FASHION_H_FC_2], dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_decision_bias_2", node=node))
    raw_ig_feature = tf.matmul(flat_data, fc_h_weights) + fc_h_bias
    ig_feature = tf.nn.relu(raw_ig_feature)
    ig_feature_size = ig_feature.get_shape().as_list()[-1]
    # OK
    hyperplane_weights = tf.Variable(
        tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="hyperplane_weights", node=node))
    # OK
    hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="hyperplane_biases", node=node))
    node.hOpsList.extend([conv_h, relu_h, pool_h])
    node.variablesSet.add(conv_h_weights)
    node.variablesSet.add(conv_h_bias)
    node.variablesSet.add(fc_h_weights)
    node.variablesSet.add(fc_h_bias)
    node.variablesSet.add(hyperplane_weights)
    node.variablesSet.add(hyperplane_biases)
    # Decisions
    network.apply_decision(node=node, branching_feature=ig_feature, hyperplane_weights=hyperplane_weights,
                           hyperplane_biases=hyperplane_biases)
    # ***************** H: Convolution Layer *****************


def leaf_func(node, network, variables=None):
    total_prev_degrees = sum(network.degreeList[0:node.depth])
    parent_F, parent_H = network.mask_input_nodes(node=node)
    parent_F_feature_size = parent_F.get_shape().as_list()[-1]
    # Parameters
    # OK
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [parent_F_feature_size,
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
    softmax_input_dim = GlobalConstants.FASHION_F_FC_2
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
    # F Operations
    flattened = tf.contrib.layers.flatten(parent_F)
    hidden_layer_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    dropped_layer_1 = tf.nn.dropout(hidden_layer_1, network.classificationDropoutKeepProb)
    hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
    dropped_layer_2 = tf.nn.dropout(hidden_layer_2, network.classificationDropoutKeepProb)
    final_feature, logits = network.apply_loss(node=node, final_feature=dropped_layer_2,
                                               softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
    # Evaluation
    node.evalDict[network.get_variable_name(name="final_eval_feature", node=node)] = final_feature
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
