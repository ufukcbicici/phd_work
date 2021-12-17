import numpy as np
import tensorflow as tf


class IfClauseLayer(tf.keras.layers.Layer):
    def _init_(self):
        super().__init__()

    # @tf.function
    def call(self, inputs, **kwargs):
        x = inputs[0]
        a = 2 * x
        b = 3 * x
        is_training = kwargs["training"]
        if is_training:
            c = a
        else:
            c = b
        return c, a, b


if __name__ == "__main__":
    x_ = tf.keras.Input(shape=[],
                        name="x_",
                        dtype=tf.int32)
    if_clause_layer = IfClauseLayer()
    c_1, a_1, b_1 = if_clause_layer([x_])

    # x_tf = tf.ones(dtype=tf.int32, shape=(1,))
    x_np = np.ones(dtype=np.int32, shape=(1,))
    c_4, a_4, b_4 = if_clause_layer([x_np], training=True)
    model = tf.keras.Model(inputs=x_, outputs=[c_1, a_1, b_1])

    print(tf.__version__)
    # print("With tf.function")
    print("training=True")
    c_2, a_2, b_2 = model(inputs=x_np, training=True)
    print(c_2)
    print(a_2)
    print(b_2)
    print("training=False")
    c_3, a_3, b_3 = model(inputs=x_np, training=False)
    print(c_3)
    print(a_3)
    print(b_3)
    print("training without tf.Model")
    print(c_4)
    print(a_4)
    print(b_4)


# import numpy as np
# import tensorflow as tf
#
#
# class IfClauseLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#     # @tf.function
#     def call(self, inputs, **kwargs):
#         x = inputs[0]
#         a = 2 * x
#         b = 10 * x
#         is_training = kwargs["training"]
#         print("Tracing  is_training={0}".format(is_training))
#         tf.print("Tf  is_training={0}".format(is_training))
#
#         if is_training:
#             c = a
#             print("Tracing  c=a")
#             tf.print("Tf c=a")
#             tf.print("c={0}".format(c))
#         else:
#             c = b
#             print("Tracing  c=b")
#             tf.print("Tf c=b")
#             tf.print("c={0}".format(c))
#         return c, a, b
#
#
# if __name__ == "__main__":
#     gpus = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#     print(tf.__version__)
#
#     with tf.device("GPU"):
#         x_ = tf.keras.Input(shape=[],
#                             name="x_",
#                             dtype=tf.int32)
#         if_clause_layer = IfClauseLayer()
#         c_1, a_1, b_1 = if_clause_layer([x_])
#         model = tf.keras.Model(inputs=x_, outputs=[c_1, a_1, b_1])
#
#         x_np = np.ones(dtype=np.int32, shape=(1, ))
#         for i in range(5):
#             print("********************************************")
#             # true_tensor = tf.convert_to_tensor(value=True)
#             c_2, a_2, b_2 = model(inputs=x_np, training=True)
#             print(c_2)
#             c_3, a_3, b_3 = model(inputs=x_np, training=False)
#             print(c_3)
#             print("********************************************")
#
