import numpy as np
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import pickle


class CowChickenDataset:
    class_names = ["Chickens", "Cows"]

    class TfDataset:
        def __init__(self, dataset_obj, iter_obj, outputs_obj, init_obj):
            self.dataset = dataset_obj
            self.iter = iter_obj
            self.outputs = outputs_obj
            self.initializer = init_obj

    # def __init__(self, data_path, test_ratio):
    #     self.dataPath = data_path
    def __init__(self, training_data=None):
        self.datasetsDict = {}
        self.isNewEpoch = False
        self.batchSize = tf.placeholder(dtype=tf.int64, name="batchSize")
        if training_data is not None:
            self.create_tf_dataset(data=training_data, data_type="training")

    def create_tf_dataset(self, data, data_type):
        file_paths = data[:, 0]
        labels = data[:, 1]
        data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        data = data.map(CowChickenDataset.process_path)
        data = data.batch(batch_size=self.batchSize)
        # if data_type == "training":
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        outputs = iterator.get_next()
        initializer = iterator.make_initializer(data)
        self.datasetsDict[data_type] = CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

    def get_next_batch(self, sess, data_type="training"):
        try:
            x = sess.run(self.datasetsDict[data_type].outputs)
            self.isNewEpoch = False
            return x
        except tf.errors.OutOfRangeError:
            self.isNewEpoch = True
            return None

    @staticmethod
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        # Read jpeg image
        img = tf.image.decode_jpeg(img, channels=3)
        # Convert into [0, 1] interval
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    @staticmethod
    def farm_data_reader(data_path):
        pattern = join(data_path, "*.jpg")
        img_files = tf.gfile.Glob(pattern)
        labels = []
        for img_file_path in img_files:
            label_exists_arr = [c_name in img_file_path for c_name in CowChickenDataset.class_names]
            assert any(label_exists_arr)
            labels.append(np.argmax(label_exists_arr))
        data_pairs = np.stack([np.array(img_files), np.array(labels)], axis=-1)
        return data_pairs


if __name__ == '__main__':
    sess = tf.Session()
    whole_data = CowChickenDataset.farm_data_reader(data_path=join("..", "data", "farm_data"))
    dataset = CowChickenDataset(training_data=whole_data)
    print("X")
    sess.run(dataset.datasetsDict["training"].initializer, feed_dict={dataset.batchSize: 256})
    X_hat = []
    while True:
        ret = dataset.get_next_batch(sess=sess)
        if ret is None:
            break
        X_hat.append(ret)
    # print("X")
