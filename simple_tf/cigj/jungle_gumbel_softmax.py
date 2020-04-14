import tensorflow as tf
import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle import Jungle
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.cigj.jungle_node import NodeType
from algorithms.info_gain import InfoGainLoss


class JungleGumbelSoftmax(Jungle):
    def __init__(self, node_build_funcs, h_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):
        self.zSampleCount = tf.placeholder(name="zSampleCount", dtype=tf.int32)
        self.unitTestList = [self.test_nan_sample_counts]
        self.prevEvalDict = {}
        self.gradAndVarsDict = {}
        self.decisionGradsDict = {}
        self.classificationGradsDict = {}
        self.gradAndVarsOp = None
        self.decisionGradsOp = None
        self.classificationGradsOp = None
        self.trainOp = None
        self.activationGrads = None
        self.activationGradsDecision = None
        self.activationGradsClassification = None
        self.probGradsDecision = None
        self.probGradsClassification = None
        self.zProbsGrads = None
        self.zSamplesGrads = None
        super().__init__(node_build_funcs, h_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.evalDict["oneHotLabelTensor"] = self.oneHotLabelTensor

    @staticmethod
    def sample_from_gumbel_softmax(probs, temperature, z_sample_count, batch_size, child_count):
        # Gumbel Softmax Entropy
        uniform = tf.distributions.Uniform(low=0.0, high=1.0)
        uniform_sample = uniform.sample(sample_shape=(batch_size, z_sample_count, child_count))
        gumbel_sample = -1.0 * tf.log(-1.0 * tf.log(uniform_sample))

        # Concrete
        log_probs = tf.log(probs)
        log_probs = tf.expand_dims(log_probs, dim=1)
        pre_transform = log_probs + gumbel_sample
        temp_divided = pre_transform / temperature
        # logits = tf.math.exp(temp_divided)
        # nominator = tf.expand_dims(tf.reduce_sum(logits, axis=2), dim=2)
        # z_samples = logits / nominator

        # ExpConcrete
        log_sum_exp = tf.expand_dims(tf.reduce_logsumexp(temp_divided, axis=2), axis=2)
        y_samples = temp_divided - log_sum_exp
        z_samples_stable = tf.exp(y_samples)

        return z_samples_stable

        # results = sess.run([gumbel_sample, uniform_sample, probs, log_probs, pre_transform,
        #                     temp_divided, logits, nominator, z_samples, y_samples, z_samples_stable],
        #                    feed_dict={x_tensor: x.samples,
        #                               batch_size_tensor: sample_count,
        #                               z_sample_count_tensor: z_sample_count,
        #                               temperature_tensor: temperature})
        # p_z = np.mean(results[2], axis=0)

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        node.H_output = branching_feature
        if node.depth < len(self.degreeList) - 1:
            node_degree = self.degreeList[node.depth + 1]
            if node_degree > 1:
                # Step 1: Create Hyperplanes
                ig_feature_size = node.H_output.get_shape().as_list()[-1]
                hyperplane_weights = UtilityFuncs.create_variable(
                    name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node),
                    shape=[ig_feature_size, node_degree],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1,
                                                    seed=GlobalConstants.SEED,
                                                    dtype=GlobalConstants.DATA_TYPE))
                hyperplane_biases = UtilityFuncs.create_variable(
                    name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node),
                    shape=[node_degree],
                    dtype=GlobalConstants.DATA_TYPE,
                    initializer=tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE))
                if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                    node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                                  momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                                  training=tf.cast(self.isTrain, tf.bool))
                # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
                activations = FastTreeNetwork.fc_layer(x=node.H_output, W=hyperplane_weights, b=hyperplane_biases,
                                                       node=node)
                node.activationsDict[node.index] = activations
                decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
                p_F_given_x = tf.nn.softmax(decayed_activation)
                p_c_given_x = self.oneHotLabelTensor
                node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                          balance_coefficient=self.informationGainBalancingCoefficient)
                # Step 3:
                # If training: Sample Z from Gumbel-Softmax distribution, based on p(F|x).
                # If testing: Pick Z = argmax_F p(F|x)
                # Prevent exactly zero probabilities by adding a constant.
                equal_mask = tf.cast(tf.equal(p_F_given_x, 0.0), tf.float32)
                mask_prob = GlobalConstants.INFO_GAIN_LOG_EPSILON * equal_mask
                p_F_given_x_corrected = p_F_given_x + mask_prob

                category_count = tf.constant(node_degree)
                z_samples = JungleGumbelSoftmax.sample_from_gumbel_softmax(probs=p_F_given_x_corrected,
                                                                           temperature=node.gumbelSoftmaxTemperature,
                                                                           z_sample_count=self.zSampleCount,
                                                                           batch_size=tf.cast(self.batchSize, tf.int32),
                                                                           child_count=node_degree)
                z_probs_matrix = tf.reduce_mean(z_samples, axis=1)
                arg_max_indices = tf.argmax(p_F_given_x_corrected, axis=1, output_type=tf.int32)
                arg_max_one_hot_matrix = tf.one_hot(arg_max_indices, category_count)
                eval_matrix = tf.where(self.evalMultipath > 0, p_F_given_x_corrected, arg_max_one_hot_matrix)
                node.conditionProbabilities = tf.where(self.isTrain > 0, z_probs_matrix, eval_matrix)
                # Reporting
                node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
                node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
                node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
                node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
                node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
                node.evalDict[UtilityFuncs.get_variable_name(name="branch_probs", node=node)] = p_F_given_x
                node.evalDict[UtilityFuncs.get_variable_name(name="branch_probs_corrected", node=node)] = p_F_given_x_corrected
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="z_samples", node=node)] = z_samples
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="z_probs_matrix", node=node)] = z_probs_matrix
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)] = arg_max_indices
                node.evalDict[
                    UtilityFuncs.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
                node.evalDict[UtilityFuncs.get_variable_name(name="eval_matrix", node=node)] = eval_matrix
            else:
                node.conditionProbabilities = tf.expand_dims(
                    tf.ones_like(tensor=self.labelTensor, dtype=tf.float32), axis=1)
            node.evalDict[
                UtilityFuncs.get_variable_name(name="conditionProbabilities", node=node)] = node.conditionProbabilities
        node.F_output = node.F_input

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                              is_train=is_train, use_masking=use_masking)
        # Set Gumbel Softmax Sample Count
        feed_dict[self.zSampleCount] = GlobalConstants.CIGJ_GUMBEL_SOFTMAX_SAMPLE_COUNT
        # Set Gumbel Softmax Temperatures at each h_nodes.
        if not self.isBaseline:
            for node in self.topologicalSortedNodes:
                if not node.nodeType == NodeType.h_node:
                    continue
                if is_train:
                    temperature = node.gumbelSoftmaxTemperatureCalculator.value
                    feed_dict[node.gumbelSoftmaxTemperature] = temperature
                    UtilityFuncs.print("{0} value={1}".format(node.gumbelSoftmaxTemperatureCalculator.name,
                                                              temperature))
                    node.gumbelSoftmaxTemperatureCalculator.update(iteration=iteration + 1)
                else:
                    feed_dict[node.gumbelSoftmaxTemperature] = GlobalConstants.CIGJ_GUMBEL_SOFTMAX_TEST_TEMPERATURE
        return feed_dict

    # Unit test methods
    def test_nan_sample_counts(self, eval_dict):
        print("mainLoss:{0}".format(eval_dict["mainLoss"]))
        print("regularizationLoss:{0}".format(eval_dict["regularizationLoss"]))
        print("decisionLoss:{0}".format(eval_dict["decisionLoss"]))
        # Check activations
        activations_dict = {k: v for k, v in eval_dict.items() if "_activations" in k}
        for k, v in activations_dict.items():
            print("{0} max:{1} min:{2}".format(k, np.max(v), np.min(v)))
        # Check for nan gradients
        for grad_var_tpl in self.gradAndVarsDict:
            if np.any(np.isnan(grad_var_tpl[0])):
                print("Gradient contains nan!")
        # Check for Gumbel-Softmax samples.
        z_samples_dict = {k: v for k, v in eval_dict.items() if "z_samples" in k}
        for k, v in z_samples_dict.items():
            nan_arr = np.isnan(v)
            if np.any(nan_arr):
                print("{0} contains nan!".format(k))
        # Check all-close inputs
        tuples = [(12, 100), (12, 95), (12, 2)]
        for tpl in tuples:
            print("np.allclose(eval_dict[\"Node1_F_input\"][{0}], eval_dict[\"Node1_F_input\"][{1}])={2}".format(
                tpl[0], tpl[1],
                np.allclose(eval_dict["Node1_F_input"][tpl[0]], eval_dict["Node1_F_input"][tpl[1]])))
            if np.allclose(eval_dict["Node1_F_input"][tpl[0]], eval_dict["Node1_F_input"][tpl[1]]):
                print("All Close.")
        # Check node1 loss weights
        node1_params = {k: v for k, v in eval_dict.items() if "l2_" in k and "Node1" in k}
        print(node1_params)
        # Check sample count arrays
        sample_count_arrays = {k: v for k, v in eval_dict.items() if "sample_count" in k}
        for k, v in sample_count_arrays.items():
            if np.isnan(v):
                print("NAN!")
        for k in eval_dict.keys():
            self.prevEvalDict[k] = eval_dict[k]

        # h_nodes = [node for node in self.topologicalSortedNodes if node.nodeType == NodeType.h_node]
        # for h_node in h_nodes:
        #     if len(self.dagObject.parent
        #     s(node=h_node)) == 1:
        #         continue
        #     parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=h_node)
        #                       if f_node.nodeType == NodeType.f_node]
        #     parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
        #     parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=h_node)
        #                       if h_node.nodeType == NodeType.h_node]
        #     assert len(parent_h_nodes) == 1
        #     f_input = eval_dict[UtilityFuncs.get_variable_name(name="F_input", node=h_node)]
        #     condition_probabilities = eval_dict[UtilityFuncs.get_variable_name(name="conditionProbabilities",
        #                                                                        node=parent_h_nodes[0])]
        #     f_outputs_prev_layer = [eval_dict[UtilityFuncs.get_variable_name(name="F_output", node=node)]
        #                             for node in parent_f_nodes]
        #     f_input_manual = np.zeros_like(f_input)
        #     for r in range(condition_probabilities.shape[0]):
        #         selected_node_idx = np.asscalar(np.argmax(condition_probabilities[r, :]))
        #         f_input_manual[r, :] = f_outputs_prev_layer[selected_node_idx][r, :]
        #     assert np.allclose(f_input, f_input_manual)
