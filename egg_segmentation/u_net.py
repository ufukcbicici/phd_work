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
        self.imageWidth = tf.placeholder(name="imageWidth", dtype=tf.int32)
        self.imageHeight = tf.placeholder(name="imageHeight", dtype=tf.int32)
        self.l2Coefficient = tf.placeholder(name="l2Coefficient", dtype=tf.float32)

    # Data augmentation if we are doing training
    def get_input(self):
        self.imageInput = tf.reshape(self.maskInput, [1, self.imageHeight, self.imageWidth, 3])
        self.maskInput = tf.reshape(self.maskInput, [1, self.imageHeight, self.imageWidth, 1])
        # Augmented Training Input
        # concat_image = tf.concat([self.imageInput, self.maskInput], axis=-1)
        # maybe_flipped = tf.image.random_flip_left_right(concat_image)
        # maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        # image = maybe_flipped[:, :, :-1]
        # mask = maybe_flipped[:, :, -1:]
        # # image = tf.image.random_brightness(image, 0.7)
        # # image = tf.image.random_hue(image, 0.3)
        # # Evaluation Input
        # final_image = tf.where(self.isTrain, image, self.imageInput)
        # final_mask = tf.where(self.isTrain, mask, self.maskInput)
        # final_image = tf.expand_dims(final_image, 0)
        # final_mask = tf.expand_dims(final_mask, 0)
        return self.imageInput, self.imageInput

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

    def upconv_2D(self, tensor, n_filter, name):
        return tf.layers.conv2d_transpose(
            tensor,
            filters=n_filter,
            kernel_size=2,
            strides=2,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2Coefficient),
            name="upsample_{}".format(name))

    def upconv_concat(self, inputA, input_B, n_filter, name):
        up_conv = self.upconv_2D(inputA, n_filter, name)

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
        # Expanding Part
        up6 = self.upconv_concat(conv5, conv4, 64, name=6)
        conv6 = self.conv_conv_pool(up6, [64, 64], name=6, pool=False)
        up7 = self.upconv_concat(conv6, conv3, 32, name=7)
        conv7 = self.conv_conv_pool(up7, [32, 32], name=7, pool=False)
        up8 = self.upconv_concat(conv7, conv2, 16, name=8)
        conv8 = self.conv_conv_pool(up8, [16, 16], name=8, pool=False)
        up9 = self.upconv_concat(conv8, conv1, 8, name=9)
        conv9 = self.conv_conv_pool(up9, [8, 8], name=9, pool=False)
        # Output
        return conv9



dataset = EggDataset()
dataset.load_dataset()
unet = UNet(dataset=dataset)
conv_net = unet.build_network()
np_img, np_msk = dataset.get_next_image()
plt.imshow(np_img)
plt.show()

# np_img, np_msk = dataset.get_next_image()
sess = tf.Session()
res = sess.run([conv_net],
               feed_dict={unet.isTrain: True, unet.imageInput: np_img, unet.maskInput: np_msk,
                          unet.imageHeight: np_img.shape[0], unet.imageWidth: np_img.shape[1]})
plt.imshow(res[0].astype(np.uint8))
plt.show()

print("X")
