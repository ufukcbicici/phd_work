import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.custom_layers.info_gain_layer import InfoGainLayer
from tf_2_cign.custom_layers.weighted_batch_norm import WeightedBatchNormalization
from tf_2_cign.utilities import Utilities


class CignScRoutingPrepLayer(tf.keras.layers.Layer):
    def __init__(self, network, level):
        super().__init__()
        self.network = network
        self.nodesInLevel = self.network.orderedNodesPerLevel[level]
        # self.FOutputs = [self.network.nodeOutputsDict[node.index]["F"] for node in self.nodesInLevel]
        # self.igMatrices = [self.network.nodeOutputsDict[node.index]["ig_mask_matrix"] for node in self.nodesInLevel]
        # self.scMasks = [self.network.scMasksDict[node.index] for node in self.nodesInLevel]

    # Tasks:
    # 1) Gather all F-outputs and ig matrix outputs of the nodes in the given level.
    # 2) Zero out of non-activated F-outputs by corresponding secondary masks.
    # 3) Output the sparse F-outputs of the whole level as a single concatenated tensor.
    # 4) This output will be used by custom functions to generate secondary routing matrices for the next level.
    @tf.function
    def call(self, inputs, **kwargs):
        f_outputs = inputs[0]
        ig_matrices = inputs[1]
        sc_masks = inputs[2]
        sparse_features_list = []
        ig_matrices_list = []
        for idx in range(len(self.nodesInLevel)):
            f_output = f_outputs[idx]
            ig_matrix = ig_matrices[idx]
            sparsity_tensor = tf.identity(sc_masks[idx])
            for _ in range(len(f_output.get_shape()) - 1):
                sparsity_tensor = tf.expand_dims(sparsity_tensor, axis=-1)
            sparsity_tensor = tf.cast(sparsity_tensor, dtype=f_output.dtype)
            sparse_features = sparsity_tensor * f_output
            sparse_features_list.append(sparse_features)
            ig_matrices_list.append(ig_matrix)
        sparse_features_complete = tf.concat(sparse_features_list, axis=-1)
        ig_matrices_complete = tf.concat(ig_matrices_list, axis=-1)
        return sparse_features_complete, ig_matrices_complete
