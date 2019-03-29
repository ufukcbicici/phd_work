import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from egg_segmentation.egg_dataset import EggDataset


class UNet:
    L2_COEFFICIENT = 0.0
    INITIAL_LR = 0.0001
    LR_SCHEDULE = [(0.00005, 10000), (0.000025, 20000), (0.00001, 30000)]
    EPOCH_COUNT = 100
    WINDOW_SIZE = 256
    STRIDE = 32
    BATCH_SIZE = 64

    def __init__(self, dataset):
        self.dataset = dataset
        self.batchSize = tf.placeholder(name="batchSize", dtype=tf.int32)
        self.isTrain = tf.placeholder(name="isTrain", dtype=tf.bool)
        self.imageInput = tf.placeholder(name="imageInput", dtype=tf.float32)
        self.maskInput = tf.placeholder(name="maskInput", dtype=tf.int32)
        self.weightInput = tf.placeholder(name="weightInput", dtype=tf.float32)
        self.imageWidth = tf.placeholder(name="imageWidth", dtype=tf.int32)
        self.imageHeight = tf.placeholder(name="imageHeight", dtype=tf.int32)
        self.l2Coefficient = tf.placeholder(name="l2Coefficient", dtype=tf.float32)
        self.logits = None
        self.loss = None
        self.optimizer = None
        self.globalCounter = None
        self.learningRate = None
        self.extraUpdateOps = None

    # Data augmentation if we are doing training
    def get_input(self):
        self.imageInput = tf.reshape(self.imageInput, [self.batchSize, UNet.WINDOW_SIZE, UNet.WINDOW_SIZE, 3])
        self.maskInput = tf.reshape(self.maskInput, [self.batchSize, UNet.WINDOW_SIZE, UNet.WINDOW_SIZE])
        self.weightInput = tf.reshape(self.weightInput, [self.batchSize, UNet.WINDOW_SIZE, UNet.WINDOW_SIZE])
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
        return self.imageInput, self.maskInput, self.weightInput

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
        tf_img, tf_msk, tf_weights = self.get_input()
        # Entry
        net = tf_img / 127.5 - 1
        # Contracting Part
        conv1, pool1 = self.conv_conv_pool(net, [8, 8], name=1)
        conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], name=2)
        conv3, pool3 = self.conv_conv_pool(pool2, [32, 32], name=3)
        conv4, pool4 = self.conv_conv_pool(pool3, [64, 64], name=4)
        conv5 = self.conv_conv_pool(pool4, [128, 128], name=5, pool=False)
        # up_conv = self.upconv_2D(conv5, 64, name=10)
        # Expanding Part
        up6 = self.upconv_concat(conv5, conv4, 64, name=6)
        conv6 = self.conv_conv_pool(up6, [64, 64], name=6, pool=False)
        up7 = self.upconv_concat(conv6, conv3, 32, name=7)
        conv7 = self.conv_conv_pool(up7, [32, 32], name=7, pool=False)
        up8 = self.upconv_concat(conv7, conv2, 16, name=8)
        conv8 = self.conv_conv_pool(up8, [16, 16], name=8, pool=False)
        up9 = self.upconv_concat(conv8, conv1, 8, name=9)
        conv9 = self.conv_conv_pool(up9, [8, 8], name=9, pool=False)
        # Logits
        self.logits = tf.layers.conv2d(conv9, self.dataset.get_label_count(), (1, 1), name='final',
                                       activation=tf.nn.relu,
                                       padding='same')
        # Loss Function
        # flat_logits = tf.reshape(logits, [-1, self.dataset.get_label_count()])
        # flat_labels = tf.reshape(tf_msk, [-1])
        # flat_weights = tf.reshape(tf_weights, [-1])
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_msk,
                                                                                   logits=self.logits)
        weighted_ce_loss_tensor = tf.multiply(cross_entropy_loss_tensor, tf_weights)
        self.loss = tf.reduce_mean(weighted_ce_loss_tensor)
        # return logits, cross_entropy_loss_tensor, weighted_ce_loss_tensor, tf_msk, self.loss

        def apply_segmentation(self):
            # Training Images
            for idx, tpl in enumerate(self.dataset.trainImages):
                print(tpl[0].shape)
                cropped_imgs, _, top_left_coords = EggDataset.get_cropped_images(image=tpl[0], mask=None,
                                                                                 window_size=UNet.WINDOW_SIZE,
                                                                                 stride=UNet.WINDOW_SIZE)
                res = sess.run([self.logits],
                               feed_dict={self.batchSize: cropped_imgs.shape[0],
                                          self.isTrain: False,
                                          self.imageInput: cropped_imgs})
                # Patch the logit image together
                # learned_mask = np.zeros_like(tpl[0])
                patched_mask = np.zeros(shape=(tpl[0].shape[0], tpl[0].shape[1]), dtype=np.int32)
                for i, yx in enumerate(top_left_coords):
                    cropped_argmax = np.argmax(res[i], axis=2)
                    patched_mask[yx[0]:yx[0]+UNet.WINDOW_SIZE, yx[1]:yx[1]+UNet.WINDOW_SIZE][:] = cropped_argmax
                print(idx)

    def build_optimizer(self):
        # Build optimizer
        self.globalCounter = tf.Variable(0, trainable=False)
        boundaries = [tpl[1] for tpl in UNet.LR_SCHEDULE]
        values = [UNet.INITIAL_LR]
        values.extend([tpl[0] for tpl in UNet.LR_SCHEDULE])
        self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
        self.extraUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extraUpdateOps):
            # self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.loss,
            #                                                                              global_step=self.globalCounter)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.globalCounter)


dataset = EggDataset(window_size=UNet.WINDOW_SIZE, stride=UNet.STRIDE)
dataset.load_dataset()
unet = UNet(dataset=dataset)
unet.build_network()
unet.build_optimizer()
# np_img, np_msk, np_weights = dataset.get_next_image(make_divisible_to=16)
# while True:
#     np_img, np_msk, np_weights = dataset.get_next_image(make_divisible_to=16)
# if np_img.shape[1] == 256:
#     break
# plt.imshow(np_img)
# plt.show()

# np_img, np_msk = dataset.get_next_image()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# [unet.optimizer, unet.loss, unet.learningRate]

for epoch_id in range(UNet.EPOCH_COUNT):
    while True:
        np_img, np_msk, np_weights = dataset.get_next_batch(batch_size=UNet.BATCH_SIZE)
        res = sess.run([unet.optimizer, unet.loss, unet.learningRate],
                       feed_dict={unet.batchSize: UNet.BATCH_SIZE,
                                  unet.isTrain: True,
                                  unet.imageInput: np_img,
                                  unet.maskInput: np_msk,
                                  unet.weightInput: np_weights})
        print("loss:{0} lr:{1}".format(res[1], res[2]))
        if dataset.isNewEpoch:
            unet.apply_segmentation()
            dataset.reset()
            break

# res = sess.run([conv_net],
#                feed_dict={unet.isTrain: True,
#                           unet.imageInput: np_img,
#                           unet.maskInput: np_msk,
#                           unet.weightInput: np_weights,
#                           unet.imageHeight: np_img.shape[0],
#                           unet.imageWidth: np_img.shape[1]})
# plt.imshow(res[0].astype(np.uint8))
# plt.show()

print("X")
