from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
import numpy as np
import tensorflow as tf


class SoftmaxCompresser:
    def __init__(self):
        pass

    @staticmethod
    def compress_network_softmax(network, sess, dataset):
        # Get all final feature vectors for all leaves, for the complete training set.
        featureVectorsDict = {}
        posteriorsDict = {}
        oneHotLabelsDict = {}
        softmaxWeights = {}
        softmaxBiases = {}
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
        while True:
            results = network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            for node in network.topologicalSortedNodes:
                if not node.isLeaf:
                    continue
                leaf_node = node
                posterior_ref = network.get_variable_name(name="posterior_probs", node=leaf_node)
                posterior_probs = results[posterior_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=posteriorsDict, key=leaf_node.index, array=posterior_probs)
                final_feature_ref = network.get_variable_name(name="final_eval_feature", node=leaf_node)
                final_leaf_features = results[final_feature_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=featureVectorsDict, key=leaf_node.index,
                                                     array=final_leaf_features)
                one_hot_label_ref = "Node{0}_one_hot_label_tensor".format(leaf_node.index)
                one_hot_labels = results[one_hot_label_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=oneHotLabelsDict, key=leaf_node.index,
                                                     array=one_hot_labels)
                softmax_weights_ref = network.get_variable_name(name="fc_softmax_weights", node=leaf_node)
                softmaxWeights[leaf_node.index] = results[softmax_weights_ref]
                softmax_biases_ref = network.get_variable_name(name="fc_softmax_biases", node=leaf_node)
                softmaxBiases[leaf_node.index] = results[softmax_biases_ref]
            if dataset.isNewEpoch:
                break
        # Train all leaf classifiers by distillation
        for node in network.topologicalSortedNodes:
            if not node.isLeaf:
                continue
            leaf_node = node
            softmax_weight = softmaxWeights[leaf_node.index]
            softmax_bias = softmaxBiases[leaf_node.index]
            feature_vectors = featureVectorsDict[leaf_node.index]
            probs = posteriorsDict[leaf_node.index]
            # Unit Test
            SoftmaxCompresser.assert_prob_correctness(softmax_weights=softmax_weight, softmax_biases=softmax_bias,
                                                      features=feature_vectors, probs=probs)
        print("X")


    @staticmethod
    def assert_prob_correctness(softmax_weights, softmax_biases, features, probs):
        logits = np.dot(features, softmax_weights) + softmax_biases
        exp_logits = np.exp(logits)
        logit_sums = np.sum(exp_logits, 1).reshape(exp_logits.shape[0], 1)
        manual_probs = exp_logits / logit_sums
        is_equal = np.allclose(probs, manual_probs)
        print("is_equal={0}".format(is_equal))
        assert is_equal