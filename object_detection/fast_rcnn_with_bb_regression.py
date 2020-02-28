import tensorflow as tf
import numpy as np
import cv2

from object_detection.constants import Constants
from object_detection.fast_rcnn import FastRcnn
from object_detection.residual_network_generator import ResidualNetworkGenerator


class FastRcnnWithBBRegression(FastRcnn):
    def __init__(self, roi_list, class_count, background_label):
        super().__init__(roi_list, class_count, background_label)
        self.bbRegressionOutputs = []
        self.bbRegressionOutputsTensor = None
        self.trueImageHeights = tf.placeholder(tf.float32, shape=(None,))
        self.bbRegressionTargets = tf.placeholder(tf.float32, shape=(None, None, 4))
        self.reshapedRegressionTargets = None
        self.selectedRegressionOutputs = None
        self.backgroundMask = None
        self.bbRegressionLoss = None
        self.weightedBbRegressionLoss = None
        self.totalPositiveSamples = None
        self.regressionLoss = None
        self.regressionLambda = tf.placeholder(tf.float32)

    def calculate_regression_targets(self, roi_labels, roi_proposals_tensor_real_coord, ground_truths):
        regression_targets = \
            np.zeros(shape=(roi_proposals_tensor_real_coord.shape[0], roi_proposals_tensor_real_coord.shape[1], 4))
        for idx in range(roi_labels.shape[0]):
            non_background_labels = roi_labels[idx] != self.backgroundLabel
            object_proposals = roi_proposals_tensor_real_coord[idx][non_background_labels]
            object_ground_truths = ground_truths[idx][non_background_labels]
            object_proposal_centers = np.stack(
                [0.5 * object_proposals[:, 1] + 0.5 * object_proposals[:, 3],
                 0.5 * object_proposals[:, 2] + 0.5 * object_proposals[:, 4]], axis=1)
            ground_truth_centers = np.stack(
                [0.5 * object_ground_truths[:, 1] + 0.5 * object_ground_truths[:, 3],
                 0.5 * object_ground_truths[:, 2] + 0.5 * object_ground_truths[:, 4]], axis=1)
            Gx = ground_truth_centers[:, 0]
            Gy = ground_truth_centers[:, 1]
            Px = object_proposal_centers[:, 0]
            Py = object_proposal_centers[:, 1]
            Gw = object_ground_truths[:, 3] - object_ground_truths[:, 1]
            Gh = object_ground_truths[:, 4] - object_ground_truths[:, 2]
            Pw = object_proposals[:, 3] - object_proposals[:, 1]
            Ph = object_proposals[:, 4] - object_proposals[:, 2]

            t_x = (Gx - Px) / Pw
            t_y = (Gy - Py) / Ph
            t_w = np.log(Gw / Pw)
            t_h = np.log(Gh / Ph)
            targets = np.stack([t_x, t_y, t_w, t_h], axis=1)
            regression_targets[idx][0:targets.shape[0]] = targets
        return regression_targets

    def build_roi_pooling(self):
        super().build_roi_pooling()
        regression_targets_shape = tf.shape(self.bbRegressionTargets)
        new_shape = tf.stack([tf.gather_nd(regression_targets_shape, [0]) *
                              tf.gather_nd(regression_targets_shape, [1]), 4], axis=0)
        self.reshapedRegressionTargets = tf.reshape(self.bbRegressionTargets, shape=new_shape)

    def build_detector_endpoint(self):
        super().build_detector_endpoint()

        # Build bounding box regression.
        net = self.roiFeatureVector
        hidden_layers = list(Constants.BB_REGRESSION_HIDDEN_LAYERS)
        for layer_id, layer_dim in enumerate(hidden_layers):
            net = tf.layers.dense(inputs=net, units=layer_dim, activation=tf.nn.relu)
        for class_id in range(self.classCount + 1):
            class_bb_regression_output = tf.layers.dense(inputs=net, units=4, activation=None)
            self.bbRegressionOutputs.append(class_bb_regression_output)
        self.bbRegressionOutputsTensor = tf.stack(self.bbRegressionOutputs, axis=1)
        # Let every sample select its output according to its label
        total_sample_count = tf.gather_nd(tf.shape(self.roiFeatureVector), [0])
        indices = tf.stack([tf.range(0, total_sample_count, 1), self.reshapedLabels], axis=1)
        self.selectedRegressionOutputs = tf.gather_nd(self.bbRegressionOutputsTensor, indices)
        # Mask out background samples
        self.backgroundMask = tf.cast(tf.not_equal(self.reshapedLabels, tf.constant(self.backgroundLabel)), "float32")
        self.bbRegressionLoss = tf.losses.huber_loss(labels=self.reshapedRegressionTargets,
                                                     predictions=self.selectedRegressionOutputs, delta=1.0,
                                                     reduction="none")
        self.weightedBbRegressionLoss = tf.expand_dims(self.backgroundMask, axis=1) * self.bbRegressionLoss
        self.totalPositiveSamples = tf.reduce_sum(self.backgroundMask)
        self.regressionLoss = self.regressionLambda * \
                              (tf.reduce_sum(self.weightedBbRegressionLoss) / self.totalPositiveSamples)

    def build_optimizer(self):
        # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
        # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
        # with tf.control_dependencies(self.extra_update_ops):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.totalLoss = self.classifierLoss + self.regularizerLoss + self.regressionLoss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.totalLoss, global_step=self.globalStep)

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
                    results = self.session.run([self.classProbabilities, self.bbRegressionOutputsTensor],
                                               feed_dict={self.backboneNetworkOutput: backbone_output,
                                                          self.roiInputs: np.expand_dims(proposal_batch, axis=0),
                                                          self.isTrain: 0})
                    class_probs = results[0]
                    regression_outputs = results[1]
                    max_probs = np.max(class_probs, axis=1)
                    predicted_classes = np.argmax(class_probs, axis=1)
                    predicted_regression_outputs = regression_outputs
                    predicted_offsets = predicted_regression_outputs[
                                        np.arange(predicted_regression_outputs.shape[0]), predicted_classes, :]
                    proposal_batch_original_size = np.copy(proposal_batch)
                    proposal_batch_original_size[:, [1, 3]] = \
                        proposal_batch_original_size[:, [1, 3]] * float(resized_img.shape[0])
                    proposal_batch_original_size[:, [0, 2]] = \
                        proposal_batch_original_size[:, [0, 2]] * float(resized_img.shape[1])
                    object_proposal_centers = np.stack(
                        [0.5 * proposal_batch_original_size[:, 0] + 0.5 * proposal_batch_original_size[:, 2],
                         0.5 * proposal_batch_original_size[:, 1] + 0.5 * proposal_batch_original_size[:, 3]], axis=1)
                    object_proposal_x = object_proposal_centers[:, 0]
                    object_proposal_y = object_proposal_centers[:, 1]
                    object_proposal_w = proposal_batch_original_size[:, 2] - proposal_batch_original_size[:, 0]
                    object_proposal_h = proposal_batch_original_size[:, 3] - proposal_batch_original_size[:, 1]
                    g_hat_x = object_proposal_w * predicted_offsets[:, 0] + object_proposal_x
                    g_hat_y = object_proposal_h * predicted_offsets[:, 1] + object_proposal_y
                    g_hat_w = object_proposal_w * np.exp(predicted_offsets[:, 2])
                    g_hat_h = object_proposal_h * np.exp(predicted_offsets[:, 3])
                    g_hat_left = np.clip(g_hat_x - 0.5 * g_hat_w, a_min=0.0, a_max=float(resized_img.shape[1]))
                    g_hat_top = np.clip(g_hat_y - 0.5 * g_hat_h, a_min=0.0, a_max=float(resized_img.shape[0]))
                    g_hat_right = np.clip(g_hat_x + 0.5 * g_hat_w, a_min=0.0, a_max=float(resized_img.shape[1]))
                    g_hat_bottom = np.clip(g_hat_y + 0.5 * g_hat_h, a_min=0.0, a_max=float(resized_img.shape[0]))
                    regressed_predictions = np.stack([g_hat_left, g_hat_top, g_hat_right, g_hat_bottom], axis=1)
                    regressed_predictions[:, [0, 2]] = regressed_predictions[:, [0, 2]] / float(resized_img.shape[1])
                    regressed_predictions[:, [1, 3]] = regressed_predictions[:, [1, 3]] / float(resized_img.shape[0])
                    predictions = np.concatenate(
                        [np.expand_dims(predicted_classes, axis=1),
                         np.expand_dims(max_probs, axis=1),
                         regressed_predictions], axis=1)
                    predictions_list.append(predictions)
                    batch_id += 1
            proposal_results = np.concatenate(predictions_list, axis=0)
            proposal_results[:, [3, 5]] = proposal_results[:, [3, 5]] * float(original_img.shape[0])
            proposal_results[:, [2, 4]] = proposal_results[:, [2, 4]] * float(original_img.shape[1])
            all_proposals.append(proposal_results)
        all_proposals = np.concatenate(all_proposals, axis=0)
        final_proposals = self.nms_algorithm(proposal_results=all_proposals)
        return final_proposals

    def train(self, dataset):
        total_losses = []
        classification_losses = []
        regularization_losses = []
        regression_losses = []
        iteration = 0
        self.saver = tf.train.Saver(max_to_keep=1000000)
        while True:
            images, roi_labels, roi_proposals, roi_proposals_tensor_real_coord, ground_truths = \
                self.get_image_batch(dataset=dataset)
            # Calculate regression targets
            regression_targets = self.calculate_regression_targets(
                roi_labels=roi_labels,
                roi_proposals_tensor_real_coord=roi_proposals_tensor_real_coord,
                ground_truths=ground_truths)
            # print("A")
            feed_dict = {self.imageInputs: images,
                         self.isTrain: 1,
                         self.l2Lambda: Constants.L2_LAMBDA,
                         self.roiInputs: roi_proposals,
                         self.roiLabels: roi_labels,
                         self.bbRegressionTargets: regression_targets,
                         self.regressionLambda: Constants.REGRESSION_LAMBDA}
            # results = sess.run([self.backboneNetworkOutput, self.roiPoolingOutput,
            #                     self.roiOutputShape, self.newRoiShape, self.detectorInput,
            #                     self.roiPoolingOutput, self.reshapedLabels,
            #                     self.logits,
            #                     self.crossEntropyLossTensors,
            #                     self.classifierLoss],
            #                    feed_dict=feed_dict)
            # results = self.session.run([self.totalLoss,
            #                             self.classProbabilities,
            #                             self.reshapedRegressionTargets,
            #                             self.bbRegressionOutputs,
            #                             self.bbRegressionOutputsTensor,
            #                             self.selectedRegressionOutputs,
            #                             self.reshapedLabels,
            #                             self.backgroundMask,
            #                             self.bbRegressionLoss,
            #                             self.weightedBbRegressionLoss,
            #                             self.totalPositiveSamples,
            #                             self.regressionLoss,
            #                             self.roiPoolingOutput],
            #                            feed_dict=feed_dict)
            results = self.session.run([self.totalLoss,
                                        self.classifierLoss,
                                        self.regularizerLoss,
                                        self.regressionLoss,
                                        self.classProbabilities,
                                        self.optimizer,
                                        self.roiPoolingOutput], feed_dict=feed_dict)
            # self.save_model(iteration=iteration)
            # print("B")
            total_losses.append(results[0])
            classification_losses.append(results[1])
            regularization_losses.append(results[2])
            regression_losses.append(results[3])
            # If this assertion fails, the the RoI pooled regions in the backbone output is smaller than
            # POOLED_WIDTH x POOLED_HEIGHT. Consider increase the size of IMG_WIDTHS contents
            assert np.sum(np.isinf(results[-1]) == True) == 0
            iteration += 1
            print("Iteration={0}".format(iteration))
            if len(total_losses) == Constants.RESULT_REPORTING_PERIOD:
                print("Total Loss:{0}".format(np.mean(np.array(total_losses))))
                print("Classification Loss:{0}".format(np.mean(np.array(classification_losses))))
                print("Regularization Loss:{0}".format(np.mean(np.array(regularization_losses))))
                print("Regression Loss:{0}".format(np.mean(np.array(regression_losses))))
                total_losses = []
                classification_losses = []
                regularization_losses = []
                regression_losses = []
            if iteration % Constants.MODEL_SAVING_PERIOD == 0:
                self.save_model(iteration=iteration)
            # self.test_roi_pooling(backbone_output=results[0], roi_pool_results=results[1], roi_proposals=roi_proposals)
