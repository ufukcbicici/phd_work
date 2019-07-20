import tensorflow as tf
import numpy as np

from algorithms.info_gain import InfoGainLoss
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants


class CignMixtureOfExperts(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset)
        GlobalConstants.USE_UNIFIED_BATCH_NORM = False

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
        node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x

    def mask_input_nodes(self, node):
        print("Masking Node:{0}".format(node.index))
        if node.isRoot:
            node.labelTensor = self.labelTensor
            node.indicesTensor = self.indicesTensor
            node.oneHotLabelTensor = self.oneHotLabelTensor
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        else:
            # Obtain the mask vector, sample counts and determine if this node receives samples.
            parent_node = self.dagObject.parents(node=node)[0]
            # Mask all inputs: F channel, H  channel, activations from ancestors, labels
            parent_F = parent_node.fOpsList[-1]
            parent_H = parent_node.hOpsList[-1]
            for k, v in parent_node.activationsDict.items():
                node.activationsDict[k] = v
            node.labelTensor = parent_node.labelTensor
            node.indicesTensor = parent_node.indicesTensor
            node.oneHotLabelTensor = parent_node.oneHotLabelTensor
            return parent_F, parent_H

    def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
        node.residueOutputTensor = final_feature
        node.finalFeatures = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
        node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
        logits = tf.matmul(final_feature, softmax_weights) + softmax_biases
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
                                                                                   logits=logits)
        pre_loss = tf.reduce_mean(cross_entropy_loss_tensor)
        loss = tf.where(tf.is_nan(pre_loss), 0.0, pre_loss)
        node.fOpsList.extend([cross_entropy_loss_tensor, pre_loss, loss])
        node.lossList.append(loss)
        return final_feature, logits

        # # Step 1: Sample random from uniform distribution. No use of information gain.
        # assert self.get_variable_name(name="sample_count", node=node) in node.evalDict
        # batch_size = node.evalDict[self.get_variable_name(name="sample_count", node=node)]
        # # During training, sample from F ~ p(F|x)
        # uniform_probs = (1.0 / float(node_degree)) * tf.ones_like(p_n_given_x)
        # sampled_indices = self.sample_from_categorical(probs=uniform_probs, batch_size=batch_size,
        #                                                category_count=tf.constant(node_degree))
        # # During testing, pick F = argmax_F p(F|x)
        # arg_max_indices = tf.argmax(p_n_given_x, axis=1, output_type=tf.int32)
        # chosen_indices = tf.where(self.isTrain > 0, sampled_indices, arg_max_indices)
        # node.evalDict[self.get_variable_name(name="branching_feature", node=node)] = branching_feature
        # node.evalDict[self.get_variable_name(name="activations", node=node)] = activations
        # node.evalDict[self.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
        # node.evalDict[self.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
        # node.evalDict[self.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
        # node.evalDict[self.get_variable_name(name="p(n|x)", node=node)] = p_n_given_x
        # node.evalDict[self.get_variable_name(name="uniform_probs", node=node)] = uniform_probs
        # node.evalDict[self.get_variable_name(name="chosen_indices", node=node)] = chosen_indices
        # # Step 4: Apply partitioning for corresponding F nodes in the same layer.
        # child_nodes = self.dagObject.children(node=node)
        # child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
        # for index in range(len(child_nodes)):
        #     child_node = child_nodes[index]
        #     child_index = child_node.index
        #     mask_tensor = tf.reshape(tf.equal(x=chosen_indices, y=tf.constant(index, tf.int32),
        #                                       name="Mask_without_threshold_{0}".format(child_index)), [-1])
        #     node.maskTensors[child_index] = mask_tensor
        #     node.evalDict[self.get_variable_name(name="mask_tensors", node=node)] = node.maskTensors
