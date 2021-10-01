import tensorflow as tf


class CignRlRoutingLayer(tf.keras.layers.Layer):
    infeasible_action_penalty = -1000000.0

    def __init__(self, level, node, network, use_ig_in_actions):
        super().__init__()
        self.level = level
        self.node = node
        self.network = network
        self.useIgInActions = use_ig_in_actions
        self.actionSpaces = [tf.constant(self.network.actionSpaces[idx])
                             for idx in range(len(self.network.actionSpaces))]
        self.reachabilityMatrices = [tf.constant(self.network.reachabilityMatrices[idx])
                                     for idx in range(len(self.network.reachabilityMatrices))]

    @tf.function
    def call(self, inputs, **kwargs):
        q_table_predicted = inputs[0]
        input_ig_routing_matrix = inputs[1]
        is_warm_up_period = inputs[2]
        past_actions = inputs[3]

        # Get the feasibility table
        feasibility_matrix = tf.gather_nd(self.reachabilityMatrices[self.level], tf.expand_dims(past_actions, axis=-1))
        # Set entries for unfeasible actions to -inf, feasible actions to 0.
        penalty_matrix = tf.where(tf.cast(feasibility_matrix, dtype=tf.bool),
                                  0.0,
                                  CignRlRoutingLayer.infeasible_action_penalty)
        # Final q table, with Bellman Optimality.
        q_table_with_penalties = penalty_matrix + q_table_predicted
        predicted_actions = tf.argmax(q_table_with_penalties, axis=-1)
        # secondary_routing_matrix = self.actionSpaces[self.level][predicted_actions]
        secondary_routing_matrix = tf.gather_nd(self.actionSpaces[self.level],
                                                tf.expand_dims(predicted_actions, axis=-1))
        secondary_routing_matrix_logical_or_ig = tf.cast(
            tf.logical_or(tf.cast(secondary_routing_matrix, dtype=tf.bool),
                          tf.cast(input_ig_routing_matrix, dtype=tf.bool)), dtype=tf.int32)

        # Actions are updated with the inclusion of information gain.
        if self.useIgInActions:
            basis_matrix = secondary_routing_matrix_logical_or_ig * \
                           tf.expand_dims(self.network.actionSpacesReverse[self.level], axis=0)
            predicted_actions = tf.reduce_sum(basis_matrix, axis=-1) - 1

        secondary_routing_matrix_warm_up = tf.ones_like(secondary_routing_matrix_logical_or_ig)

        secondary_routing_matrix_final = tf.where(tf.cast(is_warm_up_period, tf.int32) > 0,
                                                  secondary_routing_matrix_warm_up,
                                                  secondary_routing_matrix_logical_or_ig)
        return predicted_actions, secondary_routing_matrix_final
