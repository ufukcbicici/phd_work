import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
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
                img = self.get_next_batch(sess=sess, outputs=outputs)
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
        return crops, label

    def create_training_set(self, sess, X, y):
        file_paths = X
        labels = y
        data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        data = data.map(CowChickenDataset.process_path)
        self.data_preparataion(sess=sess, data=data)
        data = data.shuffle(buffer_size=file_paths.shape[0])
        data = data.map(self.augment_for_training)
        data = data.batch(batch_size=self.batchSize)
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        outputs = iterator.get_next()
        initializer = iterator.make_initializer(data)
        return CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

    def create_test_dataset(self, X):
        file_paths = X
        labels = np.zeros(shape=(X.shape[0], ), dtype=np.int32)
        data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        data = data.map(CowChickenDataset.process_path)
        data = data.map(self.augment_for_testing)
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        outputs = iterator.get_next()
        initializer = iterator.make_initializer(data)
        return CowChickenDataset.TfDataset(data, iterator, outputs, initializer)

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


class ResnetGenerator:
    @staticmethod
    def conv(name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            assert len(x.get_shape().as_list()) == 4
            assert x.get_shape().as_list()[3] == in_filters
            assert strides[1] == strides[2]
            n = filter_size * filter_size * out_filters
            shape = [filter_size, filter_size, in_filters, out_filters]
            initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n))
            kernel = tf.get_variable("conv_kernel", shape, initializer=initializer, dtype=tf.float32, trainable=True)
            x_hat = tf.nn.conv2d(x, kernel, strides, padding='SAME')
            return x_hat

    @staticmethod
    def batch_norm(name, x, is_train, momentum):
        normalized_x = tf.layers.batch_normalization(inputs=x, name=name, momentum=momentum,
                                                     training=tf.cast(is_train, tf.bool))
        return normalized_x

    @staticmethod
    def stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    @staticmethod
    def relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        if leakiness <= 0.0:
            return tf.nn.relu(features=x, name="relu")
        else:
            return tf.nn.leaky_relu(features=x, alpha=leakiness, name="leaky_relu")

    @staticmethod
    def global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    @staticmethod
    def bottleneck_residual(x, is_train, in_filter, out_filter, stride, relu_leakiness, activate_before_residual,
                            bn_momentum):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope("common_bn_relu"):
                x = ResnetGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResnetGenerator.relu(x, relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope("residual_bn_relu"):
                orig_x = x
                x = ResnetGenerator.batch_norm("init_bn", x, is_train, bn_momentum)
                x = ResnetGenerator.relu(x, relu_leakiness)

        with tf.variable_scope("sub1"):
            x = ResnetGenerator.conv("conv_1", x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope("sub2"):
            x = ResnetGenerator.batch_norm("bn2", x, is_train, bn_momentum)
            x = ResnetGenerator.relu(x, relu_leakiness)
            x = ResnetGenerator.conv("conv2", x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope("sub3"):
            x = ResnetGenerator.batch_norm("bn3", x, is_train, bn_momentum)
            x = ResnetGenerator.relu(x, relu_leakiness)
            x = ResnetGenerator.conv("conv3", x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope("sub_add"):
            if in_filter != out_filter or not all([d == 1 for d in stride]):
                orig_x = ResnetGenerator.conv("project", orig_x, 1, in_filter, out_filter, stride)
            x += orig_x
        return x

    # MultiGpu OK
    @staticmethod
    def get_input(input_net, out_filters, first_conv_filter_size, stride):
        assert input_net.get_shape().ndims == 4
        input_filters = input_net.get_shape().as_list()[-1]
        x = ResnetGenerator.conv("init_conv", input_net, first_conv_filter_size, input_filters, out_filters,
                                 ResnetGenerator.stride_arr(stride))
        return x

    # MultiGpu OK
    @staticmethod
    def get_output(x, is_train, leakiness, bn_momentum):
        x = ResnetGenerator.batch_norm("final_bn", x, is_train, bn_momentum)
        x = ResnetGenerator.relu(x, leakiness)
        x = ResnetGenerator.global_avg_pool(x)
        return x


class ResNet50Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 regularizer_coeff,
                 strides,
                 activate_before_residual,
                 features,
                 num_of_units_per_block,
                 relu_leakiness,
                 first_conv_filter_size,
                 class_count,
                 batch_size,
                 iterations):
        self.regularizer_coeff = regularizer_coeff
        self.strides = strides
        self.activate_before_residual = activate_before_residual
        self.features = features
        self.num_of_units_per_block = num_of_units_per_block
        self.relu_leakiness = relu_leakiness
        self.first_conv_filter_size = first_conv_filter_size
        self.class_count = class_count
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = tf.Session()

        assert len(self.strides) + 1 == len(self.activate_before_residual) + 1 == \
               len(self.features) == len(self.num_of_units_per_block) + 1

        # Build ResNet-50 Model; preferably with small number for features
        with tf.variable_scope("ResNet50"):
            self.inputImages = tf.placeholder(dtype=tf.float32,
                                              shape=[None,
                                                     CowChickenDataset.resnet_dimension,
                                                     CowChickenDataset.resnet_dimension,
                                                     3])
            self.inputLabels = tf.placeholder(dtype=tf.int32,
                                              shape=[None])
            self.isTrain = tf.placeholder(name="isTrain", dtype=tf.bool)
            self.globalStep = tf.Variable(0, name='global_step', trainable=False)
            # Input Block
            x = ResnetGenerator.get_input(input_net=self.inputImages, out_filters=self.features[0],
                                          first_conv_filter_size=first_conv_filter_size, stride=2)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            for block_id, unit_count in enumerate(self.num_of_units_per_block):
                for unit_id in range(unit_count):
                    with tf.variable_scope("unit_{0}{1}".format(block_id, unit_id)):
                        if unit_id == 0:
                            in_filter = self.features[block_id]
                            strd = ResnetGenerator.stride_arr(self.strides[block_id])
                            activate = self.activate_before_residual[block_id]
                        else:
                            in_filter = self.features[block_id + 1]
                            activate = False
                            strd = ResnetGenerator.stride_arr(1)
                        x = ResnetGenerator.bottleneck_residual(
                            x=x,
                            in_filter=in_filter,
                            out_filter=self.features[block_id + 1],
                            stride=strd,
                            activate_before_residual=activate,
                            relu_leakiness=self.relu_leakiness,
                            is_train=self.isTrain,
                            bn_momentum=0.9)
            # Logit Layers
            with tf.variable_scope('loss_layer'):
                self.gapOutput = ResnetGenerator.get_output(x=x, is_train=self.isTrain, leakiness=self.relu_leakiness,
                                                            bn_momentum=0.9)
                self.logits = tf.layers.dense(name="logits_fc", inputs=self.gapOutput, units=self.class_count,
                                              activation=None)
                self.posteriors = tf.nn.softmax(self.logits)

            # Loss Calculations
            self.cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputLabels,
                                                                                            logits=self.logits)
            self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy_loss_tensor)
            # L2-Weight Decay Regularization
            trainable_vars = tf.trainable_variables(scope="ResNet50")
            kernel_vars = [var for var in trainable_vars if "kernel" in var.name]
            l2_norms = [tf.nn.l2_loss(kv) for kv in kernel_vars]
            self.l2Norm = self.regularizer_coeff * tf.add_n(l2_norms)
            # Total Loss
            self.total_loss = self.cross_entropy_loss + self.l2Norm
            # Optimizer
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.total_loss, global_step=self.globalStep)

    def fit(self, X_train, y_train, X_test, y_test):
        self.sess.run(tf.global_variables_initializer())
        dataset = CowChickenDataset()
        train_data_generator = dataset.create_training_set(sess=self.sess, X=X_train, y=y_train)
        self.sess.run(train_data_generator.initializer, feed_dict={dataset.batchSize: self.batch_size})
        losses = []
        for iteration_id in range(self.iterations):
            minibatch = dataset.get_next_batch(sess=self.sess, outputs=train_data_generator.outputs)
            if minibatch is None:
                self.sess.run(train_data_generator.initializer, feed_dict={dataset.batchSize: self.batch_size})
                continue
            images = minibatch[0]
            labels = minibatch[1]
            results_dict = self.sess.run({
                "optimizer": self.optimizer,
                "loss": self.total_loss,
                "probs": self.posteriors
            }, feed_dict={
                self.inputImages: images, self.inputLabels: labels, self.isTrain: True})
            losses.append(results_dict["loss"])
            if len(losses) % 10 == 0:
                avg_loss = np.mean(losses)
                print("Iteration:{0} Loss:{1}".format(iteration_id, avg_loss))
                losses = []
            if (iteration_id + 1) % 100 == 0:
                y_train_hat = self.predict(X=X_train, dataset=dataset)
                y_test_hat = self.predict(X=X_test, dataset=dataset)
                print("Train Accuracy:{0}".format(np.mean(y_train_hat == y_train)))
                print("Test Accuracy:{0}".format(np.mean(y_test_hat == y_test)))

    def predict(self, X, dataset):
        test_data_generator = dataset.create_test_dataset(X=X)
        self.sess.run(test_data_generator.initializer, feed_dict={dataset.batchSize: 1})
        y_hat = []
        while True:
            minibatch = dataset.get_next_batch(sess=self.sess, outputs=test_data_generator.outputs)
            if minibatch is None:
                break
            test_crops = minibatch[0]
            results_dict = self.sess.run({
                "probs": self.posteriors
            }, feed_dict={
                self.inputImages: test_crops, self.isTrain: False})
            mean_prob = np.mean(results_dict["probs"], axis=0)
            y_hat.append(np.argmax(mean_prob))
        return np.array(y_hat)


if __name__ == '__main__':
    # Always get the same train-test split
    np.random.seed(67)
    # sess = tf.Session()

    train_data, test_data = CowChickenDataset.farm_data_reader(data_path=join("..", "data", "farm_data"))
    # dataset = CowChickenDataset()
    # dataset.create_tf_dataset(sess=sess, data=train_data, data_type="training")
    # dataset.create_tf_dataset(sess=sess, data=test_data, data_type="test")
    # sess.run(dataset.datasetsDict["training"].initializer, feed_dict={dataset.batchSize: 256})
    # sess.run(dataset.datasetsDict["test"].initializer, feed_dict={dataset.batchSize: 1})

    resnet50_classifier = ResNet50Classifier(
        regularizer_coeff=0.0001,
        strides=[1, 2, 2, 2],
        activate_before_residual=[True, False, False, False],
        features=[8, 32, 64, 128, 256],
        num_of_units_per_block=[3, 4, 6, 3],
        relu_leakiness=0.1,
        first_conv_filter_size=7,
        class_count=2,
        batch_size=256,
        iterations=10000)

    resnet50_classifier.fit(X_train=train_data[:, 0],
                            y_train=train_data[:, 1].astype(np.int32),
                            X_test=test_data[:, 0],
                            y_test=test_data[:, 1].astype(np.int32))

    # X_hat = []
    # y_hat = []
    # while True:
    #     minibatch = dataset.get_next_batch(sess=sess, outputs=dataset.datasetsDict["training"].outputs)
    #     if minibatch is None:
    #         break
    #     X_hat.append(minibatch[0])
    #     y_hat.append(minibatch[1].astype(np.int32))
    #
    #     test_batch = dataset.get_next_batch(sess=sess, outputs=dataset.datasetsDict["test"].outputs)
    #     print("X")

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
