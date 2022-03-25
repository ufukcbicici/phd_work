import numpy as np
import tensorflow as tf


class RoutingStrategy(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pass
