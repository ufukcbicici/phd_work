import tensorflow as tf


class CignBinaryRlRoutingLayer(tf.keras.layers.Layer):
    infeasible_action_penalty = -1000000.0

    def __init__(self, level, node, network):
        super().__init__()
        self.level = level
        self.node = node
        self.network = network

    def call(self, inputs, **kwargs):
        q_table_predicted = inputs[0]
        input_ig_routing_matrix = inputs[1]
        is_warm_up_period = inputs[2]
        ig_activations = inputs[3]
        sc_routing_matrix = inputs[4]
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
        sc_routing_matrix_2 = tf.stack(sc_mask_vectors, axis=-1)

        # Binary routing calculation algorithm:
        # If final_actions[i] == 0:
        # We will look into sc_routing_matrix[i, :] for the "current level".
        # sc_routing_matrix[i, :] contains the nodes which are activated in this level.
        # We will activate those nodes in the next layer such that:
        # 1) They are children of the nodes in the previous layer, which are indicated by the "1" entries in
        # sc_routing_matrix[i, :].
        # 2) They are the children of the nodes as indicated in "1)" but only if they are picked by the information
        # driven routing distribution in their parent nodes. Meaning i is picked if for its parent node j:
        # i = argmax_k p(n_j=k|x) for the current sample.
        # If final_actions[i] == 1:
        # We will look into sc_routing_matrix[i, :] for the "current level".
        # For every active node in this layer, we will activate "all" the children of that active node, in the
        # next layer.



        print("X")



    # @tf.function
    # def call(self, inputs, **kwargs):
    #     q_table_predicted = inputs[0]
    #     input_ig_routing_matrix = inputs[1]
    #     is_warm_up_period = inputs[2]
    #     past_actions = inputs[3]
    #
    #     # Get the feasibility table
    #     feasibility_matrix = tf.gather_nd(self.reachabilityMatrices[self.level], tf.expand_dims(past_actions, axis=-1))
    #     # Set entries for unfeasible actions to -inf, feasible actions to 0.
    #     penalty_matrix = tf.where(tf.cast(feasibility_matrix, dtype=tf.bool),
    #                               0.0,
    #                               CignRlRoutingLayer.infeasible_action_penalty)
    #     # Final q table, with Bellman Optimality.
    #     q_table_with_penalties = penalty_matrix + q_table_predicted
    #     predicted_actions = tf.argmax(q_table_with_penalties, axis=-1)
    #     # secondary_routing_matrix = self.actionSpaces[self.level][predicted_actions]
    #     secondary_routing_matrix = tf.gather_nd(self.actionSpaces[self.level],
    #                                             tf.expand_dims(predicted_actions, axis=-1))
    #     secondary_routing_matrix_logical_or_ig = tf.cast(
    #         tf.logical_or(tf.cast(secondary_routing_matrix, dtype=tf.bool),
    #                       tf.cast(input_ig_routing_matrix, dtype=tf.bool)), dtype=tf.int32)
    #
    #     # Actions are updated with the inclusion of information gain.
    #     if self.useIgInActions:
    #         basis_matrix = secondary_routing_matrix_logical_or_ig * \
    #                        tf.expand_dims(self.network.actionSpacesReverse[self.level], axis=0)
    #         predicted_actions = tf.reduce_sum(basis_matrix, axis=-1) - 1
    #
    #     secondary_routing_matrix_warm_up = tf.ones_like(secondary_routing_matrix_logical_or_ig)
    #
    #     secondary_routing_matrix_final = tf.where(tf.cast(is_warm_up_period, tf.int32) > 0,
    #                                               secondary_routing_matrix_warm_up,
    #                                               secondary_routing_matrix_logical_or_ig)
    #     return predicted_actions, secondary_routing_matrix_final
