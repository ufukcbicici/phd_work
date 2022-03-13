import unittest

import numpy as np
import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class SparseCategoricalAccuracy(unittest.TestCase):
    test_count = 100
    class_count = 10
    arr_size = 125

    def test_sparse_category_accuracy_test(self):
        lbls_all = []
        logits_all = []
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_metric")

        for test_id in range(SparseCategoricalAccuracy.test_count):
            lbl = np.random.randint(low=0, high=SparseCategoricalAccuracy.class_count,
                                    size=(SparseCategoricalAccuracy.arr_size,))
            logits = np.random.uniform(low=0.0, high=1.0, size=(SparseCategoricalAccuracy.arr_size,
                                                                SparseCategoricalAccuracy.class_count))
            low_index = np.random.randint(low=10, high=30)
            high_index = np.random.randint(low=30, high=40)
            logits[np.arange(SparseCategoricalAccuracy.arr_size), lbl] = 5.0
            logits[np.arange(start=low_index, stop=high_index), 3] = 10.0
            lbls_all.append(lbl)
            logits_all.append(logits)
            acc_metric.update_state(y_true=lbl, y_pred=logits)
        lbls_all = np.concatenate(lbls_all, axis=0)
        logits_all = np.concatenate(logits_all, axis=0)

        accuracy_gt = np.mean(lbls_all == np.argmax(logits_all, axis=1)).astype(dtype=np.float32)
        accuracy_pred = acc_metric.result().numpy()
        print("accuracy_gt={0}".format(accuracy_gt))
        print("accuracy_pred={0}".format(accuracy_pred))
        self.assertTrue(np.allclose(accuracy_gt, accuracy_pred))


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    unittest.main()
