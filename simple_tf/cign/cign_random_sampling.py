import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, FixedParameter, DecayingParameter
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.cign.cign_with_sampling import CignWithSampling


class CignRandomSample(CignWithSampling):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset)

    def apply_decision(self, node, branching_feature):
        if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
            branching_feature = tf.layers.batch_normalization(inputs=branching_feature,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain,
                                                                               tf.bool))
        ig_feature_size = node.hOpsList[-1].get_shape().as_list()[-1]
        node_degree = self.degreeList[node.depth]
        hyperplane_weights = tf.Variable(
            tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
        hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
        activations = tf.matmul(branching_feature, hyperplane_weights) + hyperplane_biases
        node.activationsDict[node.index] = activations
        decayed_activation = node.activationsDict[node.index] / node.softmaxDecay
        p_n_given_x = tf.nn.softmax(decayed_activation)
        p_c_given_x = node.oneHotLabelTensor
        node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                                  balance_coefficient=self.informationGainBalancingCoefficient)
        # Step 1: Sample random from uniform distribution. No use of information gain.
        assert self.get_variable_name(name="sample_count", node=node) in node.evalDict
        batch_size = node.evalDict[self.get_variable_name(name="sample_count", node=node)]
        # During training, sample from F ~ p(F|x)
        uniform_probs = (1.0 / float(node_degree)) * tf.ones_like(p_n_given_x)
        sampled_indices = self.sample_from_categorical(probs=uniform_probs, batch_size=batch_size,
                                                       category_count=tf.constant(node_degree))
        # During testing, pick F = argmax_F p(F|x)
        arg_max_indices = tf.argmax(p_n_given_x, axis=1, output_type=tf.int32)
        chosen_indices = tf.where(self.isTrain > 0, sampled_indices, arg_max_indices)
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        node.evalDict[self.get_variable_name(name="uniform_probs", node=node)] = uniform_probs
        node.evalDict[self.get_variable_name(name="chosen_indices", node=node)] = chosen_indices
        # Step 4: Apply partitioning for corresponding F nodes in the same layer.
        child_nodes = self.dagObject.children(node=node)
        child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
        for index in range(len(child_nodes)):
            child_node = child_nodes[index]
            child_index = child_node.index
            mask_tensor = tf.reshape(tf.equal(x=chosen_indices, y=tf.constant(index, tf.int32),
                                              name="Mask_without_threshold_{0}".format(child_index)), [-1])
            node.maskTensors[child_index] = mask_tensor
            node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors

    def set_hyperparameters(self, **kwargs):
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        if not self.isBaseline:
            GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
            GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
            self.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                    value=kwargs["decision_keep_probability"])

            # Noise Coefficient
            self.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                                decay=0.0,
                                                                decay_period=1,
                                                                min_limit=0.0)
            # Decision Loss Coefficient
            self.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                    value=0.0)
            for node in self.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                # Probability Threshold
                node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
                initial_value = 0.0
                # initial_value = 1.0 / float(node_degree)
                threshold_name = self.get_variable_name(name="prob_threshold_calculator", node=node)
                # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
                #                                                  decay_period=70000,
                #                                                  min_limit=0.4)
                node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
                # Softmax Decay
                decay_name = self.get_variable_name(name="softmax_decay", node=node)
                node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                                value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                                decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                                decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                                min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)