import tensorflow as tf


class CignBinaryRlRoutingLayer(tf.keras.layers.Layer):
    infeasible_action_penalty = -1000000.0

    def __init__(self, level, node, network):
        super().__init__()
        self.level = level
        self.node = node
        self.network = network
        self.nodeCountInThisLevel = len(self.network.orderedNodesPerLevel[level])

    def call(self, inputs, **kwargs):
        q_table_predicted = inputs[0]
        is_warm_up_period = inputs[1]
        ig_activations = inputs[2]
        sc_routing_matrix_prev_level = inputs[3]
        is_training = kwargs["training"]
        # q_table_predicted_cign_output, input_ig_routing_matrix, self.warmUpPeriodInput, ig_activations_array

        # Pick action with epsilon greedy approach
        thresholds = tf.random.uniform(shape=[q_table_predicted.shape[0]], dtype=q_table_predicted.dtype,
                                       minval=0.0, maxval=1.0)
        # Get epsilon
        eps = self.network.exploreExploitEpsilon(self.network.globalStep)
        # Determine which sample will pick exploration (eps > thresholds)
        # which will pick exploitation (thresholds >= eps)
        explore_exploit_vec = eps > thresholds
        # Argmax indices for q_table
        exploit_actions = tf.argmax(q_table_predicted, axis=-1)
        # Uniformly random indies for q_table
        explore_actions = tf.random.uniform(shape=[q_table_predicted.shape[0]], dtype=exploit_actions.dtype,
                                            minval=0, maxval=2)
        training_actions = tf.where(explore_exploit_vec, explore_actions, exploit_actions)
        test_actions = exploit_actions
        final_actions = tf.where(is_training, training_actions, test_actions)

        sc_mask_vectors = [self.network.scMaskInputsDict[node.index]
                           for node in self.network.orderedNodesPerLevel[self.level]]
        sc_routing_matrix_prev_level_2 = tf.stack(sc_mask_vectors, axis=-1)

        # Binary routing calculation algorithm:
        # If final_actions[i] == 0:
        # We will look into sc_routing_matrix_prev_level[i, :] for the "current level".
        # sc_routing_matrix_prev_level[i, :] contains the nodes which are activated in this level.
        # We will activate those nodes in the next layer such that:
        # 1) They are children of the nodes in the previous layer, which are indicated by the "1" entries in
        # sc_routing_matrix_prev_level[i, :].
        # 2) They are the children of the nodes as indicated in "1)" but only if they are picked by the information
        # driven routing distribution in their parent nodes. Meaning i is picked if for its parent node j:
        # i = argmax_k p(n_j=k|x) for the current sample.
        # If final_actions[i] == 1:
        # We will look into sc_routing_matrix_prev_level[i, :] for the "current level".
        # For every active node in this layer, we will activate "all" the children of that active node, in the
        # next layer.

        # ig_activations = tf.stack(
        #     [self.igActivationsDict[nd.index] for nd in self.orderedNodesPerLevel[level]], axis=-1)
        # ************ final_actions[i] == 0 ************
        sc_routing_matrix_action_0 = []
        for nd_idx in range(self.nodeCountInThisLevel):
            activations_nd = ig_activations[:, :, nd_idx]
            sc_routing_vector = sc_routing_matrix_prev_level[:, nd_idx]
            ig_indices = tf.argmax(activations_nd, axis=-1)
            ig_routing_matrix = tf.one_hot(ig_indices, activations_nd.shape[1])
            rl_routing_matrix = sc_routing_vector * ig_routing_matrix
            sc_routing_matrix_action_0.append(rl_routing_matrix)
        # ************ final_actions[i] == 0 ************

        # ************ final_actions[i] == 1 ************
        sc_routing_matrix_action_1 = []
        for nd_idx in range(self.nodeCountInThisLevel):
            sc_routing_vector = sc_routing_matrix_prev_level[:, nd_idx]
            rl_routing_matrix = tf.ones_like(sc_routing_matrix_action_0[nd_idx])
            rl_routing_matrix = sc_routing_vector * rl_routing_matrix
            sc_routing_matrix_action_1.append(rl_routing_matrix)
        # ************ final_actions[i] == 1 ************

        sc_routing_matrix_action_0 = tf.concat(sc_routing_matrix_action_0, axis=-1)
        sc_routing_matrix_action_1 = tf.concat(sc_routing_matrix_action_1, axis=-1)
        sc_routing_matrix = tf.where(final_actions, sc_routing_matrix_action_1, sc_routing_matrix_action_0)

        # Warm up
        sc_routing_matrix_warm_up = tf.ones_like(sc_routing_matrix)

        sc_routing_matrix_final = tf.where(tf.cast(is_warm_up_period, tf.int32) > 0,
                                           sc_routing_matrix_warm_up,
                                           sc_routing_matrix)
        return final_actions, sc_routing_matrix_final
