import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


class CignBinaryRlRoutingLayer(tf.keras.layers.Layer):

    def __init__(self, level, network):
        super().__init__()
        self.level = level
        self.network = network
        self.nodeCountInThisLevel = len(self.network.orderedNodesPerLevel[level])
        self.actionResultGeneratorLayer = CignBinaryActionResultGeneratorLayer(level=level, network=network)

    def call(self, inputs, **kwargs):
        # q_table_predicted = inputs[0]
        # is_warm_up_period = inputs[1]
        ig_activations = inputs[0]
        sc_routing_matrix_curr_level = inputs[1]
        actions = inputs[2]
        # is_training = kwargs["training"]
        # q_table_predicted_cign_output, input_ig_routing_matrix, self.warmUpPeriodInput, ig_activations_array

        # # Pick action with epsilon greedy approach
        # probs = tf.random.uniform(shape=[tf.shape(q_table_predicted)[0]], dtype=q_table_predicted.dtype,
        #                           minval=0.0, maxval=1.0)
        # # Get epsilon
        # eps = self.network.exploreExploitEpsilon(self.network.globalStep)
        # # Determine which sample will pick exploration (eps > thresholds)
        # # which will pick exploitation (thresholds >= eps)
        # explore_exploit_vec = eps > probs
        # # Argmax indices for q_table
        # exploit_actions = tf.argmax(q_table_predicted, axis=-1)
        # # Uniformly random indies for q_table
        # explore_actions = tf.random.uniform(shape=[tf.shape(q_table_predicted)[0]], dtype=exploit_actions.dtype,
        #                                     minval=0, maxval=2)
        # training_actions = tf.where(explore_exploit_vec, explore_actions, exploit_actions)
        # test_actions = exploit_actions
        # # final_actions = tf.where(is_training, training_actions, test_actions)
        # if is_training:
        #     final_actions = training_actions
        # else:
        #     final_actions = test_actions

        sc_mask_vectors = [self.network.scMaskInputsDict[node.index]
                           for node in self.network.orderedNodesPerLevel[self.level]]
        sc_routing_matrix_curr_level_2 = tf.stack(sc_mask_vectors, axis=-1)

        # Binary routing calculation algorithm:
        # If final_actions[i] == 0:
        # We will look into sc_routing_matrix_curr_level[i, :] for the "current level".
        # sc_routing_matrix_curr_level[i, :] contains the nodes which are activated in this level.
        # We will activate those nodes in the next layer such that:
        # 1) They are children of the nodes in the previous layer, which are indicated by the "1" entries in
        # sc_routing_matrix_curr_level[i, :].
        # 2) They are the children of the nodes as indicated in "1)" but only if they are picked by the information
        # driven routing distribution in their parent nodes. Meaning i is picked if for its parent node j:
        # i = argmax_k p(n_j=k|x) for the current sample.
        # If final_actions[i] == 1:
        # We will look into sc_routing_matrix_curr_level[i, :] for the "current level".
        # For every active node in this layer, we will activate "all" the children of that active node, in the
        # next layer.

        # ig_activations = tf.stack(
        #     [self.igActivationsDict[nd.index] for nd in self.orderedNodesPerLevel[level]], axis=-1)
        sc_routing_next_level_matrix_action_0, sc_routing_next_level_matrix_action_1 = \
            self.actionResultGeneratorLayer([ig_activations, sc_routing_matrix_curr_level])
        actions = tf.expand_dims(actions, axis=-1)
        sc_routing_matrix = tf.where(tf.cast(actions, dtype=tf.bool),
                                     sc_routing_next_level_matrix_action_1,
                                     sc_routing_next_level_matrix_action_0)

        # Warm up
        # sc_routing_matrix_warm_up = tf.ones_like(sc_routing_matrix)
        # sc_routing_matrix_final = tf.where(tf.cast(is_warm_up_period, tf.int32) > 0,
        #                                    sc_routing_matrix_warm_up,
        #                                    sc_routing_matrix)

        return sc_routing_matrix
