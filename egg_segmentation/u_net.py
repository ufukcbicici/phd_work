import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from egg_segmentation.egg_dataset import EggDataset


class UNet:
    L2_COEFFICIENT = 0.0

    def __init__(self, dataset):
        self.dataset = dataset
        self.isTrain = tf.placeholder(name="isTrain", dtype=tf.bool)
        self.imageInput = tf.placeholder(name="imageInput", dtype=tf.float32)
        self.maskInput = tf.placeholder(name="maskInput", dtype=tf.float32)
        self.l2Coefficient = tf.placeholder(name="l2Coefficient", dtype=tf.float32)

    # Data augmentation if we are doing training
    def get_input(self):
        # Augmented Training Input
        concat_image = tf.concat([self.imageInput, self.maskInput], axis=-1)
        maybe_flipped = tf.image.random_flip_left_right(concat_image)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]
        # image = tf.image.random_brightness(image, 0.7)
        # image = tf.image.random_hue(image, 0.3)
        # Evaluation Input
        final_image = tf.where(self.isTrain, image, self.imageInput)
        final_mask = tf.where(self.isTrain, mask, self.maskInput)
        return final_image, final_mask

    # Get conv layer
    def conv_conv_pool(self,
                       input_,
                       n_filters,
                       name,
                       pool=True,
                       activation=tf.nn.relu):
        net = input_

        with tf.variable_scope("layer{}".format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(
                    net,
                    F, (3, 3),
                    activation=None,
                    padding='same',
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2Coefficient),
                    name="conv_{}".format(i + 1))
                net = tf.layers.batch_normalization(
                    net, training=self.isTrain, name="bn_{}".format(i + 1))
                net = activation(net, name="relu{}_{}".format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(
                net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

            return net, pool

    def upconv_2D(self, tensor, n_filter, flags, name):
        return tf.layers.conv2d_transpose(
            tensor,
            filters=n_filter,
            kernel_size=2,
            strides=2,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
            name="upsample_{}".format(name))

    def upconv_concat(self, inputA, input_B, n_filter, flags, name):
        up_conv = self.upconv_2D(inputA, n_filter, flags, name)

        return tf.concat(
            [up_conv, input_B], axis=-1, name="concat_{}".format(name))

    def build_network(self):
        tf_img, tf_msk = self.get_input()
        # Entry
        net = tf_img / 127.5 - 1
        # Contracting Part
        conv1, pool1 = self.conv_conv_pool(net, [8, 8], name=1)
        conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], name=2)
        conv3, pool3 = self.conv_conv_pool(pool2, [32, 32], name=3)
        conv4, pool4 = self.conv_conv_pool(pool3, [64, 64], name=4)
        conv5 = self.conv_conv_pool(pool4, [128, 128], name=5, pool=False)


dataset = EggDataset()
dataset.load_dataset()
unet = UNet(dataset=dataset)
tf_img, tf_msk = unet.get_input()
np_img, np_msk = dataset.get_next_image()
plt.imshow(np_img)
plt.show()

# np_img, np_msk = dataset.get_next_image()
sess = tf.Session()
res = sess.run([tf_img, tf_msk], feed_dict={unet.isTrain: True, unet.imageInput: np_img, unet.maskInput: np_msk})
plt.imshow(res[0].astype(np.uint8))
plt.show()

print("X")
