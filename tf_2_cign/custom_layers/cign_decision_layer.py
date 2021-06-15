import numpy as np
import tensorflow as tf
import time

from algorithms.info_gain import InfoGainLoss
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer
from tf_2_cign.utilities import Utilities


# def apply_decision(self, node, ig_mask):
#     h_net = self.nodeOutputsDict[node.index]["H"]
#     node_degree = self.degreeList[node.depth]
#     h_net_normed = self.weightedBatchNormOps[node.index]([net, ig_mask])
#     activation_layer = CignDenseLayer(output_dim=node_degree, activation=None,
#                                       node=node, use_bias=True, name="fc_op_decision")
#     self.intermediateOps[Utilities.get_variable_name(name="fc_op_decision", node=node)] = activation_layer
#     activations = activation_layer(h_net_normed)
#     # Routing temperatures
#     activations_with_temperature = activations / self.routingTemperatures[node.index]
#     p_n_given_x = tf.nn.softmax(activations_with_temperature)
#     p_c_given_x = tf.one_hot(labels, self.classCount)
#
#     # # Calculate routing probabilities
#     # h_ig_net = tf.boolean_mask(h_net, ig_mask)
#     # h_net_normed = self.maskedBatchNormOps[node.index]([h_net, h_ig_net])
#     # activations = Cign.fc_layer(x=h_net_normed,
#     #                             output_dim=node_degree,
#     #                             activation=None,
#     #                             node=node,
#     #                             name="fc_op_decision",
#     #                             use_bias=True)
#     # # Routing temperatures
#     # activations_with_temperature = activations / self.routingTemperatures[node.index]
#     # p_n_given_x = tf.nn.softmax(activations_with_temperature)
#     # p_c_given_x = tf.one_hot(labels, self.classCount)
#     # p_n_given_x_masked = tf.boolean_mask(p_n_given_x, ig_mask)
#     # p_c_given_x_masked = tf.boolean_mask(p_c_given_x, ig_mask)
#     # self.evalDict[Utilities.get_variable_name(name="p_n_given_x", node=node)] = p_n_given_x
#     # self.evalDict[Utilities.get_variable_name(name="p_c_given_x", node=node)] = p_c_given_x
#     # self.evalDict[Utilities.get_variable_name(name="p_n_given_x_masked", node=node)] = p_n_given_x_masked
#     # self.evalDict[Utilities.get_variable_name(name="p_c_given_x_masked", node=node)] = p_c_given_x_masked
#     # # Information gain loss
#     information_gain = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x_masked,
#                                              p_c_given_x_2d=p_c_given_x_masked,
#                                              balance_coefficient=self.informationGainBalanceCoeff)
#     # self.informationGainRoutingLosses[node.index] = information_gain
#     # self.evalDict[Utilities.get_variable_name(name="information_gain", node=node)] = information_gain
#     # # Information gain based routing matrix
#     # ig_routing_matrix = tf.one_hot(tf.argmax(p_n_given_x, axis=1), node_degree, dtype=tf.int32)
#     # self.evalDict[Utilities.get_variable_name(name="ig_routing_matrix_without_mask", node=node)] = ig_routing_matrix
#     # mask_as_matrix = tf.expand_dims(ig_mask, axis=1)
#     # assert "ig_mask_matrix" not in self.nodeOutputsDict[node.index]
#     # self.nodeOutputsDict[node.index]["ig_mask_matrix"] = \
#     #     tf.cast(
#     #         tf.logical_and(tf.cast(ig_routing_matrix, dtype=tf.bool), tf.cast(mask_as_matrix, dtype=tf.bool)),
#     #         dtype=tf.int32)

class CignDecisionLayer(tf.keras.layers.Layer):
    def __init__(self, class_count):
        super().__init__()
        self.classCount = tf.constant(class_count)

    @tf.function
    def call(self, *inputs, **kwargs):
        activations = inputs[0]
        labels = inputs[1]
        temperature = inputs[2]
        balance_coefficient = inputs[3]
        weight_vector = inputs[4]
        probability_vector = tf.cast(weight_vector / tf.reduce_sum(weight_vector), dtype=activations.dtype)

        batch_size = tf.shape(activations)[0]
        node_degree = tf.shape(activations)[1]

        joint_distribution = tf.ones(shape=(batch_size, self.classCount, node_degree), dtype=activations.dtype)

        # Calculate p(x)
        joint_distribution = joint_distribution * tf.expand_dims(tf.expand_dims(probability_vector, axis=-1), axis=-1)

        # Calculate p(c|x) * p(x) = p(x,c)
        p_c_given_x = tf.one_hot(labels, self.classCount)
        joint_distribution = joint_distribution * tf.expand_dims(p_c_given_x, axis=2)

        # Calculate p(n|x,c) * p(x,c) = p(x,c,n). Note that p(n|x,c) = p(n|x) since we assume conditional independence
        activations_with_temperature = activations / temperature
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_xcn = joint_distribution * tf.expand_dims(p_n_given_x, axis=1)

        # Calculate p(c,n)
        marginal_p_cn = tf.reduce_sum(p_xcn, axis=0)
        # Calculate p(n)
        marginal_p_n = tf.reduce_sum(marginal_p_cn, axis=0)
        # Calculate p(c)
        marginal_p_c = tf.reduce_sum(marginal_p_cn, axis=1)
        # Calculate entropies
        entropy_p_cn, log_prob_p_cn = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_cn)
        entropy_p_n, log_prob_p_n = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_n)
        entropy_p_c, log_prob_p_c = InfoGainLoss.calculate_entropy(prob_distribution=marginal_p_c)
        # Calculate the information gain
        information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
        information_gain = -1.0 * information_gain
        return information_gain


if __name__ == "__main__":
    print(tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    with tf.device("GPU"):
        bs = 125
        dim = 128
        momentum = 0.9
        target_val = 5.0
        classes = 10
        nd = 2
        balance = 2.0
        temp = 1.5
        activation_layer = CignDenseLayer(output_dim=nd, activation=None,
                                          node=None, use_bias=True, name="fc_op_decision")

        # MODEL 1
        ig_layer = InfoGainLayer(class_count=classes)
        # IG with weights
        x_input = tf.keras.Input(shape=(dim,), name="x_input")
        t_ = tf.keras.Input(shape=(), name="temperature")
        l_ = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)
        w_ = tf.keras.Input(shape=(), name="weight_vector", dtype=tf.int32)

        act = activation_layer(x_input)
        ig_value = ig_layer(act, l_, t_, balance, w_)

        ig_model_1 = tf.keras.Model(inputs=[x_input, t_, l_, w_], outputs={"ig_value": ig_value})

        # MODEL 2
        x_input_2 = tf.keras.Input(shape=(dim,), name="x_input_2")
        t_2 = tf.keras.Input(shape=(), name="t_2")
        l_2 = tf.keras.Input(shape=(), name="l_2", dtype=tf.int32)
        w_2 = tf.keras.Input(shape=(), name="w_2", dtype=tf.int32)

        # Routing temperatures
        masked_x = tf.boolean_mask(x_input_2, tf.cast(w_2, tf.bool))
        masked_labels = tf.boolean_mask(l_2, tf.cast(w_2, tf.bool))
        act_2 = activation_layer(masked_x)
        activations_with_temperature = act_2 / t_2
        p_n_given_x = tf.nn.softmax(activations_with_temperature)
        p_c_given_x = tf.one_hot(masked_labels, classes)
        ig_value_2 = InfoGainLoss.get_loss(p_n_given_x_2d=p_n_given_x, p_c_given_x_2d=p_c_given_x,
                                           balance_coefficient=balance)
        ig_model_2 = tf.keras.Model(inputs=[x_input_2, t_2, l_2, w_2], outputs={"ig_value": ig_value_2})

        times_ig1 = []
        times_ig2 = []
        for i in range(10000):
            x = np.random.uniform(low=-1.0, high=1.0, size=(bs, dim))
            mask_vector = np.random.randint(low=0, high=2, size=(bs,))
            class_labels = np.random.randint(low=0, high=classes, size=(bs,))

            t0 = time.time()
            outputs_dict1 = ig_model_1([x, temp, class_labels, mask_vector], training=True)
            t1 = time.time()
            outputs_dict2 = ig_model_2([x, temp, class_labels, mask_vector], training=True)
            t2 = time.time()
            print("Iter:{0} Time IG1={1} Time IG2={2} IG1={3} IG2={4}".format(
                i, t1 - t0, t2 - t1, outputs_dict1["ig_value"], outputs_dict2["ig_value"]))
            times_ig1.append(t1 - t0)
            times_ig2.append(t2 - t1)
            assert np.allclose(outputs_dict1["ig_value"], outputs_dict2["ig_value"])
        print("IG 1 Mean Time:{0}".format(np.mean(np.array(times_ig1))))
        print("IG 2 Mean Time:{0}".format(np.mean(np.array(times_ig2))))
