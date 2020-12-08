import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter
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
    def __init__(self):
        self.datasetsDict = {}
        self.isNewEpoch = False
        self.batchSize = tf.placeholder(dtype=tf.int64, name="batchSize")
        self.meanColor = None
        self.dataEigenValues = None
        self.dataEigenVectors = None

    # Training-set wise mean subtraction & PCA
    def data_preparataion(self, sess, data):
        data_stats_file_name = "data_stats.sav"
        if os.path.exists(data_stats_file_name):
            f = open(data_stats_file_name, "rb")
            stats_dict = pickle.load(f)
            self.meanColor = stats_dict["meanColor"]
            self.dataEigenValues = stats_dict["dataEigenValues"]
            self.dataEigenVectors = stats_dict["dataEigenVectors"]
            f.close()
        else:
            iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
            outputs = iterator.get_next()
            initializer = iterator.make_initializer(data)
            sess.run(initializer)
            images = []
            while True:
                img = dataset.get_next_batch(sess=sess, outputs=outputs)
                if img is None:
                    break
                images.append(img[0])
            # Calculate mean of the dataset
            means_arr = np.stack([np.mean(img, axis=(0, 1)) for img in images], axis=0)
            self.meanColor = np.mean(means_arr, axis=0)
            # Calculate the PCA over the whole training set; like the AlexNet paper (2012)
            images_flattened = np.concatenate([np.reshape(img, newshape=(img.shape[0]*img.shape[1], 3)) for img in images],
                                              axis=0)
            cov = np.cov(images_flattened, rowvar=False)
            self.dataEigenValues, self.dataEigenVectors = np.linalg.eig(cov)
            f = open(data_stats_file_name, "wb")
            pickle.dump(
                {"meanColor": self.meanColor,
                 "dataEigenValues": self.dataEigenValues,
                 "dataEigenVectors": self.dataEigenVectors}, f)
            f.close()

    @staticmethod
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        # Read jpeg image
        img = tf.image.decode_jpeg(img, channels=3)
        # Convert into [0, 1] interval
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    @staticmethod
    def augment_for_training(img, label):
        # Follow ResNet specification; sample z~U(256,480) and resize the shorter edge according to it.
        shorter_size_length = tf.random.uniform(shape=[], minval=256, maxval=480)
        original_size = tf.shape(img)
        min_length = tf.reduce_min(original_size)
        ratio = tf.cast(shorter_size_length, tf.float32) / tf.cast(min_length, tf.float32)
        new_size = (tf.cast(ratio * tf.cast(original_size[0], tf.float32), tf.int32),
                    tf.cast(ratio * tf.cast(original_size[1], tf.float32), tf.int32))
        resized_img = tf.image.resize_images(img, new_size)
        print("X")
        return resized_img, label

    def create_tf_dataset(self, sess, data, data_type):
        file_paths = data[:, 0]
        labels = data[:, 1]
        data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        data = data.map(CowChickenDataset.process_path)
        if data_type == "training":
            self.data_preparataion(sess=sess, data=data)
            data = data.shuffle(buffer_size=file_paths.shape[0])
            # Augmentation for training
            data = data.map(CowChickenDataset.augment_for_training)
            data = data.batch(batch_size=self.batchSize)
            iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
            outputs = iterator.get_next()
            initializer = iterator.make_initializer(data)
            self.datasetsDict["training"] = CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

    def get_next_batch(self, sess, outputs):
        try:
            x = sess.run(outputs)
            self.isNewEpoch = False
            return x
        except tf.errors.OutOfRangeError:
            self.isNewEpoch = True
            return None

    @staticmethod
    def farm_data_reader(data_path, test_ratio=0.1):
        pattern = join(data_path, "*.jpg")
        img_files = tf.gfile.Glob(pattern)
        labels = []
        for img_file_path in img_files:
            label_exists_arr = [c_name in img_file_path for c_name in CowChickenDataset.class_names]
            assert any(label_exists_arr)
            labels.append(np.argmax(label_exists_arr))
        data_pairs = np.stack([np.array(img_files), np.array(labels)], axis=-1)
        train_data_, test_data_ = train_test_split(data_pairs, test_size=test_ratio)
        return train_data_, test_data_


if __name__ == '__main__':
    # Always get the same train-test split
    np.random.seed(67)
    sess = tf.Session()

    train_data, test_data = CowChickenDataset.farm_data_reader(data_path=join("..", "data", "farm_data"))
    dataset = CowChickenDataset()
    dataset.create_tf_dataset(sess=sess, data=train_data, data_type="training")
    sess.run(dataset.datasetsDict["training"].initializer, feed_dict={dataset.batchSize: 1})
    X_hat = []
    while True:
        ret = dataset.get_next_batch(sess=sess, outputs=dataset.datasetsDict["training"].outputs)
        if ret is None:
            break
        X_hat.append(ret)
    print("X")

    # print("X")
    # sess.run(dataset.datasetsDict["training"].initializer, feed_dict={dataset.batchSize: 256})
    # X_hat = []
    # while True:
    #     ret = dataset.get_next_batch(sess=sess)
    #     if ret is None:
    #         break
    #     X_hat.append(ret)
    # print("X")
