import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from sklearn.decomposition import PCA
from collections import Counter
from os import listdir
import os
from os.path import isfile, join
import pickle


class CowChickenDataset:
    class_names = ["Chickens", "Cows"]
    resnet_dimension = 224
    test_shortest_edges = [225, 256, 384, 480, 640]

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
            images_flattened = np.concatenate([
                np.reshape(img,
                           newshape=(img.shape[0] * img.shape[1], 3)) for img in images], axis=0)
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
    def resize_wrt_shortest_edge(img, target_length):
        original_size = tf.shape(img)
        min_length = tf.reduce_min([original_size[0], original_size[1]])
        ratio = tf.cast(target_length, tf.float32) / tf.cast(min_length, tf.float32)
        new_size = (tf.cast(ratio * tf.cast(original_size[0], tf.float32), tf.int32),
                    tf.cast(ratio * tf.cast(original_size[1], tf.float32), tf.int32))
        resized_img = tf.image.resize_images(img, new_size)
        return resized_img

    def augment_for_training(self, img, label):
        # Step 1: Mean subtraction and random horizontal flip
        img = img - self.meanColor
        img = tf.image.random_flip_up_down(img)
        # Step 2: Follow ResNet specification; sample z~U(256,480) and resize the shorter edge according to it.
        shorter_size_length = tf.cast(tf.random.uniform(shape=[], minval=256, maxval=480), tf.int32)
        resized_img = CowChickenDataset.resize_wrt_shortest_edge(img=img, target_length=shorter_size_length)
        # Step 3: Crop 224x224 sub image
        cropped_img = tf.random_crop(resized_img,
                                     size=[CowChickenDataset.resnet_dimension, CowChickenDataset.resnet_dimension, 3])
        # Step 4: Color augmentation with PCA
        alphas = tf.random.normal(shape=[3], mean=0.0, stddev=0.1)
        coeffs = tf.expand_dims(alphas * self.dataEigenValues, axis=-1)
        delta = tf.squeeze(tf.matmul(self.dataEigenVectors.astype(np.float32), coeffs))
        final_img = cropped_img + delta
        return final_img, label
        # return img, resized_img, cropped_img, final_img, \
        #        label, new_size, original_size, shorter_size_length, alphas, coeffs, delta

    def augment_for_testing(self, img, label):
        # Use AlexNet's 10-crop testing augmentation approach (used in original ResNet paper, too).
        img_list = []
        dim = CowChickenDataset.resnet_dimension
        img = img - self.meanColor
        for edge_size in CowChickenDataset.test_shortest_edges:
            resized_img = CowChickenDataset.resize_wrt_shortest_edge(img=img, target_length=edge_size)
            resized_img_flipped = tf.image.flip_up_down(resized_img)
            for source_img in [resized_img, resized_img_flipped]:
                source_height = tf.shape(source_img)[0]
                source_width = tf.shape(source_img)[1]
                # Top left
                tl = tf.image.crop_to_bounding_box(source_img, 0, 0, dim, dim)
                # Top right
                tr = tf.image.crop_to_bounding_box(source_img, 0, source_width - dim, dim, dim)
                # Bottom left
                bl = tf.image.crop_to_bounding_box(source_img, source_height - dim, 0, dim, dim)
                # Bottom right
                br = tf.image.crop_to_bounding_box(source_img, source_height - dim, source_width - dim, dim, dim)
                # # Center
                ct = tf.image.crop_to_bounding_box(source_img,
                                                   tf.cast(((source_height - dim) / 2), tf.int32),
                                                   tf.cast(((source_width - dim) / 2), tf.int32), dim, dim)
                img_list.extend([tl, tr, bl, br, ct])
        crops = tf.stack(img_list, axis=0)
        return crops, label, img

    def create_tf_dataset(self, sess, data, data_type):
        file_paths = data[:, 0]
        labels = data[:, 1]
        data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        data = data.map(CowChickenDataset.process_path)
        if data_type == "training":
            self.data_preparataion(sess=sess, data=data)
            data = data.shuffle(buffer_size=file_paths.shape[0])
            # Augmentation for training
            data = data.map(self.augment_for_training)
        elif data_type == "test":
            # Augmentation for testing
            data = data.map(self.augment_for_testing)
        else:
            raise NotImplementedError()
        data = data.batch(batch_size=self.batchSize)
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        outputs = iterator.get_next()
        initializer = iterator.make_initializer(data)
        self.datasetsDict[data_type] = CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

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
    dataset.create_tf_dataset(sess=sess, data=test_data, data_type="test")
    sess.run(dataset.datasetsDict["training"].initializer, feed_dict={dataset.batchSize: 256})
    sess.run(dataset.datasetsDict["test"].initializer, feed_dict={dataset.batchSize: 1})
    X_hat = []
    y_hat = []
    while True:
        minibatch = dataset.get_next_batch(sess=sess, outputs=dataset.datasetsDict["training"].outputs)
        if minibatch is None:
            break
        X_hat.append(minibatch[0])
        y_hat.append(minibatch[1].astype(np.int32))

        test_batch = dataset.get_next_batch(sess=sess, outputs=dataset.datasetsDict["test"].outputs)
        print("X")

        # cv2.imshow("img", ret[0][0])
        # cv2.waitKey(0)
        # cv2.imshow("resized_img", ret[1][0])
        # cv2.waitKey(0)
        # cv2.imshow("cropped_img", ret[2][0])
        # cv2.waitKey(0)
        # cv2.imshow("final_img", ret[3][0])
        # cv2.waitKey(0)
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
