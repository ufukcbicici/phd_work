import tensorflow as tf


class CignBinaryActionSpaceGeneratorLayer(tf.keras.layers.Layer):
    def __init__(self, level, network):
        super().__init__()
        self.level = level
        self.network = network
        self.nodeCountInThisLevel = len(self.network.orderedNodesPerLevel[level])

    def call(self, inputs, **kwargs):
        ig_activations = inputs[0]
        sc_routing_matrix_curr_level = inputs[1]
        # ************ final_actions[i] == 0 ************
        sc_routing_matrix_action_0 = []
        for nd_idx in range(self.nodeCountInThisLevel):
            activations_nd = ig_activations[:, :, nd_idx]
            sc_routing_vector = sc_routing_matrix_curr_level[:, nd_idx]
            ig_indices = tf.argmax(activations_nd, axis=-1)
            ig_routing_matrix = tf.one_hot(ig_indices, activations_nd.shape[1], dtype=tf.int32)
            rl_routing_matrix = tf.expand_dims(sc_routing_vector, axis=-1) * ig_routing_matrix
            sc_routing_matrix_action_0.append(rl_routing_matrix)
        # ************ final_actions[i] == 0 ************

        # ************ final_actions[i] == 1 ************
        sc_routing_matrix_action_1 = []
        for nd_idx in range(self.nodeCountInThisLevel):
            sc_routing_vector = sc_routing_matrix_curr_level[:, nd_idx]
            rl_routing_matrix = tf.ones_like(sc_routing_matrix_action_0[nd_idx])
            rl_routing_matrix = tf.expand_dims(sc_routing_vector, axis=-1) * rl_routing_matrix
            sc_routing_matrix_action_1.append(rl_routing_matrix)
        # ************ final_actions[i] == 1 ************

        sc_routing_matrix_action_0 = tf.concat(sc_routing_matrix_action_0, axis=-1)
        sc_routing_matrix_action_1 = tf.concat(sc_routing_matrix_action_1, axis=-1)

        return [sc_routing_matrix_action_0, sc_routing_matrix_action_1]
