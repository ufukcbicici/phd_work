import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_space_generator_layer import CignBinaryActionSpaceGeneratorLayer


class CignBinaryActionGeneratorLayer(tf.keras.layers.Layer):

    def __init__(self, network):
        super().__init__()
        self.network = network

    def get_actions(self, q_table_predicted):
        # Pick action with epsilon greedy approach
        probs = tf.random.uniform(shape=[tf.shape(q_table_predicted)[0]],
                                  dtype=q_table_predicted.dtype,
                                  minval=0.0,
                                  maxval=1.0)
        global_step = self.network.globalStep
        # Get epsilon
        eps = self.network.exploreExploitEpsilon(global_step)
        # Determine which sample will pick exploration (eps > thresholds)
        # which will pick exploitation (thresholds >= eps)
        explore_exploit_vec = eps > probs
        # Argmax indices for q_table
        exploit_actions = tf.argmax(q_table_predicted, axis=-1)
        # Uniformly random indies for q_table
        explore_actions = tf.random.uniform(shape=[tf.shape(q_table_predicted)[0]], dtype=exploit_actions.dtype,
                                            minval=0, maxval=2)
        return explore_actions, exploit_actions, explore_exploit_vec

    def call(self, inputs, **kwargs):
        q_table_predicted = inputs
        # global_step = inputs[1]
        # # is_warm_up_period = inputs[1]
        # ig_activations = inputs[1]
        # sc_routing_matrix_curr_level = inputs[2]
        is_training = kwargs["training"]
        explore_actions, exploit_actions, explore_exploit_vec = self.get_actions(q_table_predicted=q_table_predicted)
        training_actions = tf.where(explore_exploit_vec, explore_actions, exploit_actions)
        test_actions = exploit_actions
        if is_training:
            final_actions = training_actions
        else:
            final_actions = test_actions
        return final_actions, explore_exploit_vec, explore_actions, exploit_actions

