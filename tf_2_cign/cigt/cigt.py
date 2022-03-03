import tensorflow as tf
import numpy as np


class Cigt(tf.keras.Model):
    def __init__(self, input_dims, class_count, path_counts, blocks_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathCounts = path_counts
        self.blocksList = blocks_list
        self.classCount = class_count
        self.inputs = tf.keras.Input(shape=input_dims, name="inputs")
        self.labels = tf.keras.Input(shape=(), name="labels", dtype=tf.int32)

