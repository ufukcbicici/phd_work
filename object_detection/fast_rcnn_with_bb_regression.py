import tensorflow as tf
import numpy as np

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

    def train(self, dataset):
        losses = []
        iteration = 0
        self.saver = tf.train.Saver()
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
                         self.bbRegressionTargets: regression_targets}
            # results = sess.run([self.backboneNetworkOutput, self.roiPoolingOutput,
            #                     self.roiOutputShape, self.newRoiShape, self.detectorInput,
            #                     self.roiPoolingOutput, self.reshapedLabels,
            #                     self.logits,
            #                     self.crossEntropyLossTensors,
            #                     self.classifierLoss],
            #                    feed_dict=feed_dict)
            results = self.session.run([self.totalLoss,
                                        self.classProbabilities,
                                        self.reshapedRegressionTargets,
                                        self.bbRegressionOutputs,
                                        self.bbRegressionOutputsTensor,
                                        self.selectedRegressionOutputs,
                                        self.reshapedLabels,
                                        self.roiPoolingOutput],
                                       feed_dict=feed_dict)
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
