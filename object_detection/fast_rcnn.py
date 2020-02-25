import tensorflow as tf
import numpy as np
import cv2
import os
from algorithms.roi_pooling import RoIPooling
from object_detection.constants import Constants
from object_detection.object_detection_data_manager import ObjectDetectionDataManager
from object_detection.residual_network_generator import ResidualNetworkGenerator
from object_detection.utilities import Utilities
from tensorflow.contrib.framework.python.framework import checkpoint_utils


class FastRcnn:
    def __init__(self, roi_list, class_count, background_label, backbone_type="ResNet"):
        self.imageInputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
        self.roiInputs = tf.placeholder(tf.float32, shape=(None, None, 4))
        self.roiLabels = tf.placeholder(tf.float32, shape=(None, None))
        self.reshapedLabels = None
        self.roiList = roi_list
        self.isTrain = tf.placeholder(name="is_train_flag", dtype=tf.int64)
        self.backboneType = backbone_type
        self.backboneNetworkOutput = None
        self.roiPoolingOutput = None
        self.roiOutputShape = None
        self.newRoiShape = None
        self.detectorInput = None
        self.detectorEndPoint = None
        self.roiFeatureVector = None
        self.classCount = class_count
        self.backgroundLabel = background_label
        self.logits = None
        self.classProbabilities = None
        self.crossEntropyLossTensors = None
        self.regularizerLoss = tf.constant(0.0)
        self.l2Lambda = tf.placeholder(tf.float32)
        self.classifierLoss = None
        self.optimizer = None
        self.totalLoss = None
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        self.backboneUpdateOps = None
        self.detectorUpdateOps = None
        self.roiPoolingInfMask = None
        self.session = tf.Session()
        self.saver = None

    def build_network(self):
        # Build the backbone
        with tf.variable_scope("Backbone_Network"):
            if self.backboneType == "ResNet":
                self.backboneNetworkOutput = self.build_resnet_backbone()
                # self.backboneUpdateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(self.backboneUpdateOps):
                #     self.backboneNetworkOutput = tf.identity(self.backboneNetworkOutput)
            else:
                raise NotImplementedError()
        # Build the roi pooling phase
        with tf.variable_scope("RoI_Pooling"):
            self.build_roi_pooling()
        with tf.variable_scope("Detector_Endpoint"):
            self.build_detector_endpoint()
        with tf.variable_scope("Optimizer"):
            self.build_optimizer()
        self.session.run(tf.initialize_all_variables())

    def build_resnet_backbone(self):
        # ResNet Parameters
        x = ResidualNetworkGenerator.generate_resnet_blocks(
            input_net=self.imageInputs,
            num_of_units_per_block=Constants.NUM_OF_RESIDUAL_UNITS,
            num_of_feature_maps_per_block=Constants.NUM_OF_FEATURES_PER_BLOCK,
            first_conv_filter_size=Constants.FIRST_CONV_FILTER_SIZE,
            relu_leakiness=Constants.RELU_LEAKINESS,
            stride_list=Constants.FILTER_STRIDES,
            active_before_residuals=Constants.ACTIVATE_BEFORE_RESIDUALS,
            is_train_tensor=self.isTrain,
            batch_norm_decay=Constants.BATCH_NORM_DECAY)
        return x

    def build_roi_pooling(self):
        # with tf.control_dependencies([self.backboneNetworkOutput]):
        self.roiPoolingOutput = RoIPooling.roi_pool(x=[self.backboneNetworkOutput, self.roiInputs],
                                                    pooled_height=Constants.POOLED_HEIGHT,
                                                    pooled_width=Constants.POOLED_WIDTH)
        # In case there inf entries in the pooling output, due to the area is smaller than
        # POOLED_WIDTH x POOLED_HEIGHT
        self.roiPoolingInfMask = tf.is_inf(self.roiPoolingOutput)
        self.roiPoolingOutput = tf.where(self.roiPoolingInfMask,
                                         tf.zeros_like(self.roiPoolingOutput),
                                         self.roiPoolingOutput)
        self.roiOutputShape = tf.shape(self.roiPoolingOutput)
        self.newRoiShape = tf.stack(
            [tf.gather_nd(self.roiOutputShape, [0]) * tf.gather_nd(self.roiOutputShape, [1]),
             self.roiPoolingOutput.get_shape().as_list()[2],
             self.roiPoolingOutput.get_shape().as_list()[3],
             self.roiPoolingOutput.get_shape().as_list()[4]], axis=0)
        self.detectorInput = tf.reshape(self.roiPoolingOutput, shape=self.newRoiShape)
        labels_shape = tf.shape(self.roiLabels)
        self.reshapedLabels = tf.reshape(self.roiLabels,
                                         shape=[tf.gather_nd(labels_shape, [0]) * tf.gather_nd(labels_shape, [1])])

    def build_detector_endpoint(self):
        x = ResidualNetworkGenerator.generate_resnet_blocks(
            input_net=self.detectorInput,
            num_of_units_per_block=Constants.DETECTOR_NUM_OF_RESIDUAL_UNITS,
            num_of_feature_maps_per_block=Constants.DETECTOR_NUM_OF_FEATURES_PER_BLOCK,
            first_conv_filter_size=Constants.DETECTOR_FIRST_CONV_FILTER_SIZE,
            relu_leakiness=Constants.DETECTOR_RELU_LEAKINESS,
            stride_list=Constants.DETECTOR_FILTER_STRIDES,
            active_before_residuals=Constants.DETECTOR_ACTIVATE_BEFORE_RESIDUALS,
            is_train_tensor=self.isTrain,
            batch_norm_decay=Constants.BATCH_NORM_DECAY)
        self.detectorEndPoint = x
        # all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # self.detectorUpdateOps = [op for op in all_update_ops if op not in set(self.backboneUpdateOps)]
        # with tf.control_dependencies(self.detectorUpdateOps):
        #     self.detectorEndPoint = tf.identity(self.detectorEndPoint)
        self.roiFeatureVector = ResidualNetworkGenerator.global_avg_pool(self.detectorEndPoint)
        # MLP for detection
        hidden_layers = list(Constants.CLASSIFIER_HIDDEN_LAYERS)
        hidden_layers.append(self.classCount)
        net = self.roiFeatureVector
        for layer_id, layer_dim in enumerate(hidden_layers):
            if layer_id < len(hidden_layers) - 1:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
            else:
                net = tf.layers.dense(inputs=net, units=layer_dim, activation=None)
        self.logits = net
        self.classProbabilities = tf.nn.softmax(self.logits)
        self.crossEntropyLossTensors = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.reshapedLabels, 'int32'),
                                                           logits=self.logits)
        self.classifierLoss = tf.reduce_mean(self.crossEntropyLossTensors)
        self.build_l2_lambda_loss()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.totalLoss = self.classifierLoss + self.regularizerLoss

    def build_l2_lambda_loss(self):
        vars = tf.trainable_variables()
        for v in vars:
            if "kernel" in v.name:
                self.regularizerLoss += self.l2Lambda * tf.nn.l2_loss(v)

    def build_optimizer(self):
        # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
        # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
        # with tf.control_dependencies(self.extra_update_ops):
        self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)

    def get_checkpoint_path(self, iteration):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.abspath(
            os.path.join(os.path.join(curr_path, "saved_model"),
                         "checkpoint_{0}_iteration_{1}".format(Constants.MODEL_NAME, iteration)))
        checkpoint_path = os.path.abspath(os.path.join(directory_path, "model.ckpt"))
        return directory_path, checkpoint_path

    def save_model(self, iteration):
        directory_path, checkpoint_path = self.get_checkpoint_path(iteration=iteration)
        os.mkdir(directory_path)
        self.saver.save(self.session, checkpoint_path)

    def load_model(self, iteration):
        _, checkpoint_path = self.get_checkpoint_path(iteration=iteration)
        saved_vars = checkpoint_utils.list_variables(checkpoint_dir=checkpoint_path)
        all_vars = tf.global_variables()
        for var in all_vars:
            if "Adam" in var.name:
                continue
            source_array = checkpoint_utils.load_variable(checkpoint_dir=checkpoint_path, name=var.name)
            tf.assign(var, source_array).eval(session=self.session)

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

    def get_image_batch(self, dataset):
        images, roi_proposals_tensor = dataset.create_image_batch(
            batch_size=Constants.IMAGE_COUNT_PER_BATCH,
            roi_sample_count=Constants.ROI_SAMPLE_COUNT_PER_IMAGE,
            positive_sample_ratio=Constants.POSITIVE_SAMPLE_RATIO_PER_IMAGE)
        roi_labels = roi_proposals_tensor[:, :, 0].astype(np.int32)
        roi_proposals = roi_proposals_tensor[:, :, 1:]
        return images, roi_labels, roi_proposals

    def calculate_accuracy_on_image(self, img, ground_truth_list):
        final_proposals = self.detect_single_image(original_img=img)
        # Convert ground truths to actual rectangles
        ground_truth_list[:, [0, ]]


        print("X")

    def detect_single_image(self, original_img):
        all_proposals = []
        for img_width in Constants.IMG_WIDTHS:
            new_width = img_width  # int(img_obj.imgArr.shape[1] * scale_percent / 100)
            new_height = int((float(img_width) / original_img.shape[1]) * original_img.shape[0])
            resized_img = cv2.resize(original_img, (new_width, new_height))
            roi_scale = img_width / min(Constants.IMG_WIDTHS)
            roi_list = (roi_scale * self.roiList).astype(np.int32)
            # Run the backbone first for this scale
            feed_dict = {self.imageInputs: np.expand_dims(resized_img, axis=0),
                         self.isTrain: 0}
            backbone_output = self.session.run([self.backboneNetworkOutput], feed_dict=feed_dict)[0]
            # Create proposals
            proposals_list = []
            for idx, roi in enumerate(roi_list):
                proposals = []
                left = 0
                while left + roi[0] <= resized_img.shape[1]:
                    top = 0
                    right = left + roi[0]
                    while top + roi[1] <= resized_img.shape[0]:
                        bottom = top + roi[1]
                        proposals.append(np.array([left, top, right, bottom]))
                        top = top + Constants.STRIDE_HEIGHT
                    left = left + Constants.STRIDE_WIDTH
                    # ObjectDetectionDataManager.print_img_with_final_rois(
                    #     img_name="Proposals_{0}.png".format(idx),
                    #     img=resized_img, roi_matrix=proposals,
                    #     colors=np.random.uniform(low=0, high=255, size=(len(proposals), 3)))
                proposals = np.stack(proposals, axis=0).astype(np.float32)
                proposals[:, [1, 3]] = proposals[:, [1, 3]] / float(resized_img.shape[0])
                proposals[:, [0, 2]] = proposals[:, [0, 2]] / float(resized_img.shape[1])
                proposals_list.append(proposals)
                # ObjectDetectionDataManager.print_img_with_final_rois(
                #     img_name="Proposals_{0}.png".format(idx),
                #     img=resized_img, roi_matrix=proposals, colors=[(0, 255, 0)] * proposals.shape[0])

            # Send proposals for classification and regression
            predictions_list = []
            for proposals in proposals_list:
                batch_id = 0
                while True:
                    proposal_batch = proposals[batch_id * Constants.TEST_BATCH_SIZE:
                                               (batch_id + 1) * Constants.TEST_BATCH_SIZE]
                    if np.prod(proposal_batch.shape) == 0:
                        break
                    results = self.session.run([self.classProbabilities],
                                               feed_dict={self.backboneNetworkOutput: backbone_output,
                                                          self.roiInputs: np.expand_dims(proposal_batch, axis=0),
                                                          self.isTrain: 0})
                    class_probs = results[0]
                    max_probs = np.max(class_probs, axis=1)
                    predicted_classes = np.argmax(class_probs, axis=1)
                    predictions = np.concatenate(
                        [np.expand_dims(predicted_classes, axis=1),
                         np.expand_dims(max_probs, axis=1),
                         proposal_batch], axis=1)
                    predictions_list.append(predictions)
                    batch_id += 1
            proposal_results = np.concatenate(predictions_list, axis=0)
            proposal_results[:, [3, 5]] = proposal_results[:, [3, 5]] * float(original_img.shape[0])
            proposal_results[:, [2, 4]] = proposal_results[:, [2, 4]] * float(original_img.shape[1])
            all_proposals.append(proposal_results)
        all_proposals = np.concatenate(all_proposals, axis=0)
        final_proposals = self.nms_algorithm(proposal_results=all_proposals)
        return final_proposals

    def nms_algorithm(self, proposal_results):
        # Eliminate all entries with background label
        non_background_proposal_indices = proposal_results[:, 0] != self.backgroundLabel
        object_proposals = proposal_results[non_background_proposal_indices]
        iou_matrix = np.apply_along_axis(lambda x: Utilities.get_iou_with_list(x, object_proposals[:, 2:]),
                                         axis=1, arr=object_proposals[:, 2:])
        final_proposals = []
        while object_proposals.shape[0] > 0:
            most_confident_id = np.argmax(object_proposals[:, 1], axis=0)
            final_proposals.append(object_proposals[most_confident_id])
            iou_distances = iou_matrix[most_confident_id, :]
            non_overlapping_flags = iou_distances < Constants.NMS_THRESHOLD
            object_proposals = object_proposals[non_overlapping_flags]
            iou_matrix = iou_matrix[non_overlapping_flags, :]
            iou_matrix = iou_matrix[:, non_overlapping_flags]
        final_proposals = np.stack(final_proposals, axis=0)
        return final_proposals

    def train(self, dataset):
        losses = []
        iteration = 0
        self.saver = tf.train.Saver()
        while True:
            images, roi_labels, roi_proposals = self.get_image_batch(dataset=dataset)
            # print("A")
            feed_dict = {self.imageInputs: images,
                         self.isTrain: 1,
                         self.l2Lambda: Constants.L2_LAMBDA,
                         self.roiInputs: roi_proposals,
                         self.roiLabels: roi_labels}
            # results = sess.run([self.backboneNetworkOutput, self.roiPoolingOutput,
            #                     self.roiOutputShape, self.newRoiShape, self.detectorInput,
            #                     self.roiPoolingOutput, self.reshapedLabels,
            #                     self.logits,
            #                     self.crossEntropyLossTensors,
            #                     self.classifierLoss],
            #                    feed_dict=feed_dict)
            results = self.session.run([self.totalLoss, self.classProbabilities,
                                        self.optimizer, self.roiPoolingOutput], feed_dict=feed_dict)
            # self.save_model(iteration=iteration)
            # print("B")
            losses.append(results[0])
            # If this assertion fails, the the RoI pooled regions in the backbone output is smaller than
            # POOLED_WIDTH x POOLED_HEIGHT. Consider increase the size of IMG_WIDTHS contents
            assert np.sum(np.isinf(results[-1]) == True) == 0
            iteration += 1
            print("Iteration={0}".format(iteration))
            if len(losses) == Constants.RESULT_REPORTING_PERIOD:
                print("Loss:{0}".format(np.mean(np.array(losses))))
                losses = []
            if iteration % Constants.MODEL_SAVING_PERIOD == 0:
                self.save_model(iteration=iteration)
            # self.test_roi_pooling(backbone_output=results[0], roi_pool_results=results[1], roi_proposals=roi_proposals)

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
