import tensorflow as tf
import numpy as np
import cv2
import os
from algorithms.roi_pooling import RoIPooling
from object_detection.constants import Constants
from object_detection.fast_rcnn import FastRcnn
from object_detection.object_detection_data_manager import ObjectDetectionDataManager
from object_detection.residual_network_generator import ResidualNetworkGenerator
from object_detection.utilities import Utilities
from tensorflow.contrib.framework.python.framework import checkpoint_utils


class FastRcnnBBRegression(FastRcnn):
    def __init__(self, roi_list, class_count, background_label, backbone_type="ResNet"):
        super().__init__(roi_list, class_count, background_label, backbone_type)

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
