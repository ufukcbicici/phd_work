import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_no_stitch import JungleNoStitch
from simple_tf.global_params import GlobalConstants
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.info_gain import InfoGainLoss


class JungleGumbelSoftmax(JungleNoStitch):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)
        self.zSampleCount = tf.placeholder(name="zSampleCount", dtype=tf.int32)
        # self.unitTestList = [self.test_stitching]

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
        node_degree = self.degreeList[node.depth + 1]
        if node_degree > 1:
            # Step 1: Create Hyperplanes
            ig_feature_size = node.H_output.get_shape().as_list()[-1]
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
            if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain, tf.bool))
            # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
            activations = tf.matmul(node.H_output, hyperplane_weights) + hyperplane_biases
            node.activationsDict[node.index] = activations
            decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
            p_F_given_x = tf.nn.softmax(decayed_activation)
            p_c_given_x = self.oneHotLabelTensor
            node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                      balance_coefficient=self.informationGainBalancingCoefficient)
            # Step 3:
            # If training: Sample Z from Gumbel-Softmax distribution, based on p(F|x).
            # If testing: Pick Z = argmax_F p(F|x)
            category_count = tf.constant(node_degree)
            z_samples = JungleGumbelSoftmax.sample_from_gumbel_softmax(probs=p_F_given_x,
                                                                       temperature=node.gumbelSoftmaxTemperature,
                                                                       z_sample_count=self.zSampleCount,
                                                                       batch_size=self.batchSize,
                                                                       child_count=node_degree)
            z_probs_matrix = tf.reduce_mean(z_samples, axis=1)
            arg_max_indices = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
            arg_max_one_hot_matrix = tf.one_hot(arg_max_indices, category_count)
            node.conditionProbabilities = tf.where(self.isTrain > 0, z_probs_matrix, arg_max_one_hot_matrix)
            # Reporting
            node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
            node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
            node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
            node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
            node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
            node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
            node.evalDict[
                UtilityFuncs.get_variable_name(name="z_samples", node=node)] = z_samples
            node.evalDict[
                UtilityFuncs.get_variable_name(name="z_probs_matrix", node=node)] = z_probs_matrix
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_indices", node=node)] = arg_max_indices
            node.evalDict[
                UtilityFuncs.get_variable_name(name="arg_max_one_hot_matrix", node=node)] = arg_max_one_hot_matrix
        else:
            node.conditionProbabilities = tf.ones_like(tensor=self.labelTensor, dtype=tf.float32)
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
