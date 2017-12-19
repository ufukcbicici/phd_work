import tensorflow as tf
from simple_tf.global_params import GlobalConstants

FIRST_LAYER = 256
SECOND_LAYER = 128
THIRD_LAYER = 64


def baseline(node, network, variables=None):
    # Parameters - Convolution Layers
    network.mask_input_nodes(node=node)
    flat_x = tf.contrib.layers.flatten(network.dataTensor)
    flat_dimension_size = flat_x.get_shape().as_list()[-1]
    # FC 1 Weights
    fc_weights_1 = tf.Variable(tf.truncated_normal(
        [flat_dimension_size, FIRST_LAYER],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_residue_weights_1", node=node))
    fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[FIRST_LAYER], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_residue_biases_1", node=node))
    # FC 2 Weights
    fc_weights_2 = tf.Variable(tf.truncated_normal(
        [FIRST_LAYER, SECOND_LAYER],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_residue_weights_2", node=node))
    fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[SECOND_LAYER], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_residue_biases_2", node=node))
    # FC 3 Weights
    fc_weights_3 = tf.Variable(tf.truncated_normal(
        [SECOND_LAYER, THIRD_LAYER],
        stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_residue_weights_3", node=node))
    fc_biases_3 = tf.Variable(tf.constant(0.1, shape=[THIRD_LAYER], dtype=GlobalConstants.DATA_TYPE),
                              name=network.get_variable_name(name="fc_residue_biases_3", node=node))
    # Softmax Weights
    fc_softmax_weights = tf.Variable(
        tf.truncated_normal([THIRD_LAYER, GlobalConstants.NUM_LABELS],
                            stddev=0.1,
                            seed=GlobalConstants.SEED,
                            dtype=GlobalConstants.DATA_TYPE),
        name=network.get_variable_name(name="fc_softmax_weights", node=node))
    fc_softmax_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
                                                dtype=GlobalConstants.DATA_TYPE),
                                    name=network.get_variable_name(name="fc_softmax_biases", node=node))
    # Fully Connected Layers
    hidden_layer_1 = tf.nn.relu(tf.matmul(flat_x,         fc_weights_1) + fc_biases_1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, fc_weights_2) + fc_biases_2)
    hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, fc_weights_3) + fc_biases_3)
    # logits = tf.matmul(hidden_layer_2, fc_softmax_weights) + fc_softmax_biases
    # Loss
    final_feature, logits = network.apply_loss(node=node, final_feature=hidden_layer_3,
                                               softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
    # Evaluation
    node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor
    # Variables Set
    node.variablesSet = {fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2, fc_weights_3, fc_biases_3,
                         fc_softmax_weights, fc_softmax_biases}

    # # Convolution Layers
    # network.mask_input_nodes(node=node)
    # # First Conv Layer
    # conv1 = tf.nn.conv2d(network.dataTensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # # Second Conv Layer
    # conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    # relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # # Third Conv Layer
    # conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
    # relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    # pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # flattened = tf.contrib.layers.flatten(pool3)
    # flat_dimension_size = flattened.get_shape().as_list()[-1]
    # # FC 1 Weights
    # fc_weights_1 = tf.Variable(tf.truncated_normal(
    #     [flat_dimension_size, GlobalConstants.FASHION_FC_1],
    #     stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
    #     name=network.get_variable_name(name="fc_weights_1", node=node))
    # fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_FC_1], dtype=GlobalConstants.DATA_TYPE),
    #                           name=network.get_variable_name(name="fc_biases_1", node=node))
    # # FC 2 Weights
    # fc_weights_2 = tf.Variable(tf.truncated_normal(
    #     [GlobalConstants.FASHION_FC_1, GlobalConstants.FASHION_FC_2],
    #     stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
    #     name=network.get_variable_name(name="fc_weights_2", node=node))
    # fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.FASHION_FC_2], dtype=GlobalConstants.DATA_TYPE),
    #                           name=network.get_variable_name(name="fc_biases_2", node=node))
    # # Softmax Weights
    # fc_softmax_weights = tf.Variable(
    #     tf.truncated_normal([GlobalConstants.FASHION_FC_2, GlobalConstants.NUM_LABELS],
    #                         stddev=0.1,
    #                         seed=GlobalConstants.SEED,
    #                         dtype=GlobalConstants.DATA_TYPE),
    #     name=network.get_variable_name(name="fc_softmax_weights", node=node))
    # fc_softmax_biases = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS],
    #                                             dtype=GlobalConstants.DATA_TYPE),
    #                                 name=network.get_variable_name(name="fc_softmax_biases", node=node))
    # # Fully Connected Layers
    # hidden_layer_1 = tf.nn.relu(tf.matmul(flattened, fc_weights_1) + fc_biases_1)
    # dropped_layer_1 = tf.nn.dropout(hidden_layer_1, network.classificationDropoutKeepProb)
    # hidden_layer_2 = tf.nn.relu(tf.matmul(dropped_layer_1, fc_weights_2) + fc_biases_2)
    # dropped_layer_2 = tf.nn.dropout(hidden_layer_2, network.classificationDropoutKeepProb)
    # # logits = tf.matmul(hidden_layer_2, fc_softmax_weights) + fc_softmax_biases
    # # Loss
    # final_feature, logits = network.apply_loss(node=node, final_feature=dropped_layer_2,
    #                                            softmax_weights=fc_softmax_weights, softmax_biases=fc_softmax_biases)
    # # Evaluation
    # node.evalDict[network.get_variable_name(name="posterior_probs", node=node)] = tf.nn.softmax(logits)
    # node.evalDict[network.get_variable_name(name="labels", node=node)] = node.labelTensor
    # # Variables Set
    # node.variablesSet = {conv1_weights, conv1_biases, conv2_weights, conv2_biases, conv3_weights, conv3_biases,
    #                      fc_weights_1, fc_biases_1, fc_weights_2, fc_biases_2, fc_softmax_weights, fc_softmax_biases}


def grad_func(network):
    # self.initOp = tf.global_variables_initializer()
    # sess.run(self.initOp)
    vars = tf.trainable_variables()
    decision_vars_list = []
    classification_vars_list = []
    residue_vars_list = []
    regularization_vars_list = []
    for v in vars:
        classification_vars_list.append(v)
        if not ("gamma" in v.name or "beta" in v.name):
            regularization_vars_list.append(v)
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
    network.decisionGradients = None
    network.residueGradients = tf.gradients(ys=network.residueLoss, xs=residue_vars_list)
    network.regularizationGradients = tf.gradients(ys=network.regularizationLoss, xs=regularization_vars_list)


def threshold_calculator_func(network):
    pass


def residue_network_func(network):
    pass


def tensorboard_func(network):
    pass