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
    def __init__(self, training_files=None):
        self.datasetsDict = {}
        self.isNewEpoch = False
        if training_files is not None:
            self.datasetsDict["training"] = self.create_tf_dataset(files_list=training_files)

    def create_tf_dataset(self, files_list):
        data = tf.data.Dataset.from_tensor_slices((files_list, ))
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        outputs = iterator.get_next()
        initializer = iterator.make_initializer(data)
        return CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

    def get_next_batch(self, sess, data_type="training"):
        try:
            x = sess.run(self.datasetsDict[data_type].outputs)
            self.isNewEpoch = False
            return x
        except tf.errors.OutOfRangeError:
            self.isNewEpoch = True
            return None

    # Get label
    @staticmethod
    def get_label_from_path(file_path):
        does_label_exist_in_path = [c_name in file_path for c_name in CowChickenDataset.class_names]
        if np.sum(does_label_exist_in_path) == 0:
            return -1
        label = np.argmax(does_label_exist_in_path)
        return label

    @staticmethod
    def process_path(file_path):
        label = CowChickenDataset.get_label_from_path(file_path=file_path)
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
        return np.array(img_files)


if __name__ == '__main__':
    sess = tf.Session()
    X = CowChickenDataset.farm_data_reader(data_path=join("..", "data", "farm_data"))
    dataset = CowChickenDataset(training_files=X)
    sess.run(dataset.datasetsDict["training"].initializer, feed_dict={})
    X_hat = []
    while True:
        ret = dataset.get_next_batch(sess=sess)
        if ret is None:
            break
        X_hat.append(ret)
    print("X")
