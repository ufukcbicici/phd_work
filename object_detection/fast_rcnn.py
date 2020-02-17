import tensorflow as tf
import numpy as np

from algorithms.roi_pooling import RoIPooling
from object_detection.constants import Constants
from object_detection.residual_network_generator import ResidualNetworkGenerator


class FastRcnn:
    def __init__(self, backbone_type="ResNet"):
        self.imageInputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
        self.roiInputs = tf.placeholder(tf.float32, shape=(None, None, 4))
        self.isTrain = tf.placeholder(name="is_train_flag", dtype=tf.int64)
        self.backboneType = backbone_type
        self.backboneNetworkOutput = None
        self.roiPoolingOutput = None

    def build_network(self):
        # Build the backbone
        with tf.variable_scope("Backbone_Network"):
            if self.backboneType == "ResNet":
                self.backboneNetworkOutput = self.build_resnet_backbone()
            else:
                raise NotImplementedError()
        # Build the roi pooling phase
        self.build_roi_pooling()

    def build_resnet_backbone(self):
        # ResNet Parameters
        num_of_units_per_block = Constants.NUM_OF_RESIDUAL_UNITS
        num_of_feature_maps_per_block = Constants.NUM_OF_FEATURES_PER_BLOCK
        first_conv_filter_size = Constants.FIRST_CONV_FILTER_SIZE
        relu_leakiness = Constants.RELU_LEAKINESS
        stride_list = Constants.FILTER_STRIDES
        active_before_residuals = Constants.ACTIVATE_BEFORE_RESIDUALS

        assert len(num_of_feature_maps_per_block) == len(stride_list) + 1 and \
               len(num_of_feature_maps_per_block) == len(active_before_residuals) + 1
        # Input layer
        x = ResidualNetworkGenerator.get_input(input=self.imageInputs, out_filters=num_of_feature_maps_per_block[0],
                                               first_conv_filter_size=first_conv_filter_size)
        # Loop over blocks, the resnet trunk
        for block_id in range(len(num_of_feature_maps_per_block) - 1):
            with tf.variable_scope("block_{0}_0".format(block_id)):
                x = ResidualNetworkGenerator.bottleneck_residual(
                    x=x,
                    in_filter=num_of_feature_maps_per_block[block_id],
                    out_filter=num_of_feature_maps_per_block[block_id + 1],
                    stride=ResidualNetworkGenerator.stride_arr(stride_list[block_id]),
                    activate_before_residual=active_before_residuals[block_id],
                    relu_leakiness=relu_leakiness,
                    is_train=self.isTrain,
                    bn_momentum=Constants.BATCH_NORM_DECAY)
            for i in range(num_of_units_per_block - 1):
                with tf.variable_scope("block_{0}_{1}".format(block_id, i + 1)):
                    x = ResidualNetworkGenerator.bottleneck_residual(
                        x=x,
                        in_filter=num_of_feature_maps_per_block[block_id + 1],
                        out_filter=num_of_feature_maps_per_block[block_id + 1],
                        stride=ResidualNetworkGenerator.stride_arr(1),
                        activate_before_residual=False,
                        relu_leakiness=relu_leakiness,
                        is_train=self.isTrain,
                        bn_momentum=Constants.BATCH_NORM_DECAY)
        return x

    def build_roi_pooling(self):
        with tf.control_dependencies([self.backboneNetworkOutput]):
            self.roiPoolingOutput = RoIPooling.roi_pool(x=[self.backboneNetworkOutput, self.roiInputs],
                                                        pooled_height=Constants.POOLED_HEIGHT,
                                                        pooled_width=Constants.POOLED_WIDTH)

    def test_roi_pooling(self, backbone_output, roi_pool_results, roi_proposals):
        pooled_imgs = []
        pooled_height = Constants.POOLED_HEIGHT
        pooled_width = Constants.POOLED_WIDTH
        for img_idx in range(backbone_output.shape[0]):
            feature_map = backbone_output[img_idx]
            img_rois = roi_proposals[img_idx]
            pooled_maps = []
            for roi_idx in range(img_rois.shape[0]):
                roi = img_rois[roi_idx]
                feature_map_height = int(feature_map.shape[0])
                feature_map_width = int(feature_map.shape[1])
                h_start = int(feature_map_height * roi[1])
                w_start = int(feature_map_width * roi[0])
                h_end = int(feature_map_height * roi[3])
                w_end = int(feature_map_width * roi[2])
                region = feature_map[h_start:h_end, w_start:w_end, :]
                # Divide the region into non overlapping areas
                region_height = h_end - h_start
                region_width = w_end - w_start
                h_step = int(region_height / pooled_height)
                w_step = int(region_width / pooled_width)
                pooled_map = np.zeros(shape=(pooled_height, pooled_width, region.shape[-1]))
                for i in range(pooled_height):
                    delta_h = h_step if i != pooled_height - 1 else region_height - i * h_step
                    for j in range(pooled_width):
                        delta_w = w_step if j != pooled_width - 1 else region_width - j * w_step
                        sub_region = region[i * h_step:i * h_step + delta_h, j * w_step:j * w_step + delta_w, :]
                        if np.prod(sub_region.shape) == 0:
                            max_val = 0.0
                        else:
                            max_val = np.max(sub_region, axis=(0, 1))
                        pooled_map[i, j, :] = max_val
                pooled_maps.append(pooled_map)
            pooled_maps = np.stack(pooled_maps, axis=0)
            pooled_imgs.append(pooled_maps)
        np_result = np.stack(pooled_imgs, axis=0)
        assert np.allclose(roi_pool_results, np_result)
        print("Test Passed.")

    def train(self, dataset):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        while True:
            # Extract sample images and roi proposals per image.
            images, roi_proposals_tensor = dataset.create_image_batch(
                batch_size=Constants.IMAGE_COUNT_PER_BATCH,
                roi_sample_count=Constants.ROI_SAMPLE_COUNT_PER_IMAGE,
                positive_sample_ratio=Constants.POSITIVE_SAMPLE_RATIO_PER_IMAGE)

            roi_labels = roi_proposals_tensor[:, :, 0].astype(np.int32)
            roi_proposals = roi_proposals_tensor[:, :, 1:]

            feed_dict = {self.imageInputs: images,
                         self.isTrain: 1,
                         self.roiInputs: roi_proposals}
            results = sess.run([self.backboneNetworkOutput, self.roiPoolingOutput], feed_dict=feed_dict)
            assert np.sum(np.isinf(results[1]) == True) == 0
            self.test_roi_pooling(backbone_output=results[0], roi_pool_results=results[1], roi_proposals=roi_proposals)
            print("X")

# net = imageInputs
# in_filters = imageInputs.get_shape().as_list()[-1]
# out_filters = 32
# pooled_height = 7
# pooled_width = 7
#
# W = tf.get_variable("W", [3, 3, in_filters, out_filters], trainable=True)
# b = tf.get_variable("b", [out_filters], trainable=True)
# net = tf.nn.conv2d(net, W, strides = [1, 2, 2, 1], padding='SAME')
# net = tf.nn.bias_add(net, b)
# net = tf.nn.relu(net)
#
# X = np.random.uniform(low=0.0, high=1.0, size=(3, 2500, 640, 3))
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# results = sess.run([net], feed_dict={imageInputs: X})
#
# print("X")
