import os
import cv2
import numpy as np
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from collections import Counter

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from sklearn.preprocessing import StandardScaler

from object_detection.utilities import Utilities


class DetectionImage(object):
    def __init__(self, img_name, img_arr):
        self.imgName = img_name
        self.imgArr = img_arr
        self.roiMatrix = None
        self.imageScales = {}


class ObjectDetectionDataManager(object):
    def __init__(self, path):
        self.dataPath = path
        self.dataList = None
        self.trainingImageIndices = None
        self.testImageIndices = None
        self.medoidRois = None
        self.backgroundLabel = None
        self.maxHeight = None

    def read_data(self):
        onlyfiles = [f for f in listdir(self.dataPath) if isfile(join(self.dataPath, f))]
        img_files = [f for f in onlyfiles if ".JPG" in f or ".jpg" in f]
        txt_files = [f for f in onlyfiles if ".txt" in f]
        assert len(img_files) == len(txt_files)
        self.dataList = []
        data_dict = {}
        # Read images
        for f in img_files:
            filename, file_extension = os.path.splitext(f)
            file_path = os.path.join(self.dataPath, f)
            img_arr = cv2.imread(filename=file_path)
            assert img_arr is not None and len(img_arr.shape) == 3 and img_arr.shape[0] > 0 and img_arr.shape[1] > 0
            img_obj = DetectionImage(img_name=filename, img_arr=img_arr)
            assert filename not in data_dict
            data_dict[filename] = img_obj
        img_widths = np.array([img_obj.imgArr.shape[1] for img_obj in data_dict.values()])
        img_heights = np.array([img_obj.imgArr.shape[0] for img_obj in data_dict.values()])
        # Read ROIs
        for f in txt_files:
            filename, file_extension = os.path.splitext(f)
            file_path = os.path.join(self.dataPath, f)
            roi_file = open(file_path, 'r')
            list_of_rois = roi_file.readlines()
            list_of_rois = [line.replace("\n", "") for line in list_of_rois]
            roi_arr = []
            for roi_str in list_of_rois:
                roi_parts = roi_str.split(" ")
                assert len(roi_parts) == 5
                roi_tpl = np.array([float(x) for x in roi_parts])
                roi_arr.append(roi_tpl)
            roi_arr = np.stack(roi_arr, axis=0)
            assert filename in data_dict
            data_dict[filename].roiMatrix = roi_arr
        self.dataList = np.array(list(data_dict.values()))

    def process_data(self, iou_threshold, max_coverage, max_num_of_medoids, test_ratio=0.15):
        # Create image scales and split into training and test sets.
        for img_obj in self.dataList:
            for img_width in Constants.IMG_WIDTHS:
                new_width = img_width  # int(img_obj.imgArr.shape[1] * scale_percent / 100)
                new_height = int((float(img_width) / img_obj.imgArr.shape[1]) * img_obj.imgArr.shape[0])
                img_obj.imageScales[img_width] = cv2.resize(img_obj.imgArr, (new_width, new_height))
        indices = np.array(range(self.dataList.shape[0]))
        self.trainingImageIndices, self.testImageIndices = train_test_split(indices, test_size=test_ratio)
        # Cluster Bounding Boxes
        training_objects = self.dataList[self.trainingImageIndices]
        self.medoidRois = BBClustering.run(training_objects=training_objects,
                                           iou_threshold=iou_threshold,
                                           max_medoid_count=max_num_of_medoids,
                                           max_coverage=max_coverage)
        self.calculate_label_distribution()

    def calculate_label_distribution(self):
        labels = [obj.roiMatrix[:, 0] for obj in self.dataList]
        labels = np.concatenate(labels, axis=0)
        c = Counter(labels)
        assert all([idx in c for idx in range(len(c))])
        assert len(c) not in c
        self.backgroundLabel = len(c)

    def save_processed_data(self):
        pickle_out_file = open(os.path.join(self.dataPath, "processed_dataset.sav"), "wb")
        pickle.dump(self, pickle_out_file)
        pickle_out_file.close()

    @staticmethod
    def load_processed_data(data_path):
        pickle_in_file = open(os.path.join(data_path, "processed_dataset.sav"), "rb")
        dataset = pickle.load(pickle_in_file)
        pickle_in_file.close()
        return dataset

    def sample_rois(self, medoids, img, true_height, roi_matrix,
                    positive_roi_count, negative_roi_count):
        #  Get medoid distribution
        found_positive_rois = 0
        found_negative_rois = 0
        total_rois = positive_roi_count + negative_roi_count
        selected_medoids = medoids[np.random.choice(medoids.shape[0], total_rois)]
        img_width = img.shape[1]
        img_height = true_height
        medoid_scale = img_width / min(Constants.IMG_WIDTHS)
        selected_medoids = medoid_scale * selected_medoids
        roi_centers = np.stack([0.5 * roi_matrix[:, 1] + 0.5 * roi_matrix[:, 3],
                                0.5 * roi_matrix[:, 2] + 0.5 * roi_matrix[:, 4]], axis=1)
        roi_labels = roi_matrix[:, 0]
        roi_centers = np.concatenate([roi_labels[:, np.newaxis], roi_centers], axis=1)
        # Select positive regions: Put every medoid (anchor) at the center of every positive bounding box. Then
        # translate them by random (Gaussian) amounts.
        # Keep the ones as positive which have IoU with the bounding box larger than > 0.5
        std = medoid_scale * Constants.POSITIVE_PROPOSAL_SAMPLING_STD
        repeat_count = 1 * medoids.shape[0]
        positive_proposals_all = None
        negative_proposals_all = None
        while True:
            # Calculate new roi centers
            proposal_centers = np.repeat(roi_centers, repeats=repeat_count, axis=0)
            noise = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.array([[std, 0.0], [0.0, std]]),
                                                  size=(proposal_centers.shape[0],))
            proposal_centers[:, 1:] = proposal_centers[:, 1:] + noise
            repeated_medoids = medoid_scale * np.concatenate([medoids] *
                                                             int(proposal_centers.shape[0] / medoids.shape[0]), axis=0)
            assert proposal_centers.shape[0] == repeated_medoids.shape[0]
            # left_coords = proposal_centers[:, 0] - 0.5 * repeated_medoids[:, 0]
            # left_coords = np.clip(left_coords, a_min=0.0, a_max=img_width)
            proposals = np.stack(
                [proposal_centers[:, 0],
                 np.clip(proposal_centers[:, 1] - 0.5 * repeated_medoids[:, 0], a_min=0.0, a_max=img_width),
                 np.clip(proposal_centers[:, 2] - 0.5 * repeated_medoids[:, 1], a_min=0.0, a_max=img_height),
                 np.clip(proposal_centers[:, 1] + 0.5 * repeated_medoids[:, 0], a_min=0.0, a_max=img_width),
                 np.clip(proposal_centers[:, 2] + 0.5 * repeated_medoids[:, 1], a_min=0.0, a_max=img_height)],
                axis=1)
            iou_matrix = np.apply_along_axis(lambda x: Utilities.get_iou_with_list(x, roi_matrix[:, 1:5]),
                                             axis=1, arr=proposals[:, 1:5])
            max_ious = np.max(iou_matrix, axis=1)
            positive_roi_indices = np.nonzero(max_ious >= Constants.POSITIVE_IOU_THRESHOLD)[0]
            positive_proposals = proposals[positive_roi_indices]
            if positive_proposals_all is None:
                positive_proposals_all = positive_proposals
            else:
                positive_proposals_all = np.concatenate([positive_proposals_all, positive_proposals], axis=0)
            if positive_proposals_all.shape[0] >= positive_roi_count:
                # bb_matrix = np.concatenate([positive_proposals_all, roi_matrix[:, 1:5]], axis=0).astype(np.int32)
                # colors = [(255, 0, 0)] * positive_proposals_all.shape[0]
                # colors.extend([(0, 0, 255)] * roi_matrix.shape[0])
                # ObjectDetectionDataManager.print_img_with_final_rois(img_name="Exp3", img=img, roi_matrix=bb_matrix,
                #                                                      colors=colors)
                break
        # Select negative regions
        sample_count = total_rois
        while True:
            left_coord_upper_limits = img_width - selected_medoids[:, 0]
            roi_left_coords = np.random.uniform(low=0, high=left_coord_upper_limits, size=sample_count)
            roi_right_coords = roi_left_coords + selected_medoids[:, 0]
            top_coord_upper_limits = img_height - selected_medoids[:, 1]
            roi_top_coords = np.random.uniform(low=0, high=top_coord_upper_limits, size=sample_count)
            roi_bottom_coords = roi_top_coords + selected_medoids[:, 1]
            proposals = np.stack([roi_left_coords, roi_top_coords, roi_right_coords, roi_bottom_coords], axis=1)
            # calculated_rois = np.stack(
            #     [sampled_rois[:, 2] - sampled_rois[:, 0], sampled_rois[:, 3] - sampled_rois[:, 1]], axis=1)
            # assert np.allclose(selected_medoids, calculated_rois)
            # Check if colludes with ground truths in the images
            iou_matrix = np.apply_along_axis(lambda x: Utilities.get_iou_with_list(x, roi_matrix[:, 1:5]),
                                             axis=1, arr=proposals)
            max_ious = np.max(iou_matrix, axis=1)
            # positive_roi_indices = np.nonzero(max_ious >= Constants.POSITIVE_IOU_THRESHOLD)[0]
            negative_roi_indices = np.nonzero(max_ious < Constants.NEGATIVE_IOU_THRESHOLD)[0]
            # positive_proposals = proposals[positive_roi_indices]
            negative_proposals = proposals[negative_roi_indices]
            negative_proposals = np.concatenate(
                [self.backgroundLabel * np.ones(shape=(negative_proposals.shape[0], 1)), negative_proposals], axis=1)
            # positive_proposals_all = np.concatenate([positive_proposals_all, positive_proposals], axis=0)
            if negative_proposals_all is None:
                negative_proposals_all = negative_proposals
            else:
                negative_proposals_all = np.concatenate([negative_proposals_all, negative_proposals], axis=0)
            if negative_proposals_all.shape[0] >= negative_roi_count:
                # bb_matrix = np.concatenate([positive_proposals_all, roi_matrix[:, 1:5]], axis=0).astype(np.int32)
                # colors = [(255, 0, 0)] * positive_proposals_all.shape[0]
                # colors.extend([(0, 0, 255)] * roi_matrix.shape[0])
                # ObjectDetectionDataManager.print_img_with_final_rois(img_name="Exp3", img=img, roi_matrix=bb_matrix,
                #                                                      colors=colors)
                break

        # Build the roi batch
        # First: Sample positive rois

        def assert_func(bb_arr):
            assert np.sum(bb_arr[:, 0] < 0) == 0 and \
                   np.sum(bb_arr[:, 1] < 0) == 0 and \
                   np.sum(bb_arr[:, 2] > img_width) == 0 and \
                   np.sum(bb_arr[:, 3] > img_height) == 0

        positive_proposals_selected_idx = np.random.choice(positive_proposals_all.shape[0], positive_roi_count,
                                                           replace=False)
        negative_proposals_selected_idx = np.random.choice(negative_proposals_all.shape[0], negative_roi_count,
                                                           replace=False)
        positive_proposals_selected = positive_proposals_all[positive_proposals_selected_idx]
        assert_func(positive_proposals_selected[:, 1:5])
        negative_proposals_selected = negative_proposals_all[negative_proposals_selected_idx]
        assert_func(negative_proposals_selected[:, 1:5])
        all_proposals = np.concatenate([positive_proposals_selected, negative_proposals_selected], axis=0)
        # Visualize
        # bb_matrix = np.concatenate([positive_proposals_selected[:, 1:5],
        #                             negative_proposals_selected[:, 1:5],
        #                             roi_matrix[:, 1:5]],
        #                            axis=0).astype(np.int32)
        # colors = [(255, 0, 0)] * positive_proposals_selected.shape[0]
        # colors.extend([(0, 0, 255)] * negative_proposals_selected.shape[0])
        # colors.extend([(255, 0, 255)] * negative_proposals_selected.shape[0])
        # ObjectDetectionDataManager.print_img_with_final_rois(img_name="Exp4", img=img, roi_matrix=bb_matrix,
        #                                                      colors=colors)
        return all_proposals

    def create_image_batch(self, batch_size, roi_sample_count, positive_sample_ratio):
        positive_roi_count = int(roi_sample_count * positive_sample_ratio)
        negative_roi_count = roi_sample_count - positive_roi_count
        # Sample scales
        selected_scale = np.asscalar(np.random.choice(np.array(Constants.IMG_WIDTHS), size=1))
        # Select images
        selected_image_indices = np.random.choice(self.trainingImageIndices, size=batch_size)
        selected_img_objects = self.dataList[selected_image_indices]
        # Zero - pad images accordingly and pack them into a single numpy array
        max_height = max([img.imageScales[selected_scale].shape[0] for img in selected_img_objects])
        assert len(set([img.imageScales[selected_scale].shape[1] for img in selected_img_objects])) == 1
        channel_count = set([img.imageScales[selected_scale].shape[2] for img in selected_img_objects])
        assert len(list(channel_count)) == 1
        images = np.zeros(shape=(batch_size, max_height, selected_scale, list(channel_count)[0]),
                          dtype=selected_img_objects[0].imageScales[selected_scale].dtype)
        roi_matrices = []
        roi_proposals_tensor = []
        label_matrix = []
        for idx in range(batch_size):
            img_height = selected_img_objects[idx].imageScales[selected_scale].shape[0]
            images[idx, 0:img_height, :, :] = selected_img_objects[idx].imageScales[selected_scale]
            # Calculate actual roi bounding box sizes
            roi_matrix = selected_img_objects[idx].roiMatrix
            reshaped_roi_matrix = np.copy(roi_matrix)
            # Left Coords
            reshaped_roi_matrix[:, 1] = (roi_matrix[:, 1] - 0.5 * roi_matrix[:, 3]) * selected_scale
            # Top Coords: Include the coefficient coming from the zero padding as well.
            reshaped_roi_matrix[:, 2] = (roi_matrix[:, 2] - 0.5 * roi_matrix[:, 4]) * float(img_height)
            # Right Coords
            reshaped_roi_matrix[:, 3] = (roi_matrix[:, 1] + 0.5 * roi_matrix[:, 3]) * selected_scale
            # Bottom Coords:Include the coefficient coming from the zero padding as well.
            reshaped_roi_matrix[:, 4] = (roi_matrix[:, 2] + 0.5 * roi_matrix[:, 4]) * float(img_height)
            reshaped_roi_matrix = np.copy(reshaped_roi_matrix).astype(np.int32)
            roi_matrices.append(reshaped_roi_matrix)
            # ********** This is for visualizing **********
            # ObjectDetectionDataManager.print_img_with_final_rois(img_name="roi_img_{0}".format(idx), img=images[idx],
            #                                                      roi_matrix=reshaped_roi_matrix)
            # self.show_image(img_obj=selected_img_objects[idx], scale=selected_scale)
            # ********** This is for visualizing **********

            # ********** This is for visualizing **********
            # y = 0
            # for roi in self.medoidRois:
            #     roi_scale = selected_scale / min(Constants.IMG_WIDTHS)
            #     scaled_roi = roi_scale * roi
            #     cv2.rectangle(images[idx],
            #                   pt1=(int(selected_scale / 2.0) - int(scaled_roi[0] / 2.0), y),
            #                   pt2=(int(selected_scale / 2.0) + int(scaled_roi[0] / 2.0), y + int(scaled_roi[1])),
            #                   color=(0, 0, 255),
            #                   thickness=3)
            #     y += int(scaled_roi[1])
            #     print("X")
            # cv2.imwrite("Image{0}.png".format(idx), images[idx])
            # ********** This is for visualizing **********
            roi_proposals = self.sample_rois(medoids=self.medoidRois, img=images[idx], true_height=img_height,
                                             roi_matrix=reshaped_roi_matrix,
                                             positive_roi_count=positive_roi_count,
                                             negative_roi_count=negative_roi_count)
            roi_proposals_tensor.append(roi_proposals)
        roi_proposals_tensor = np.stack(roi_proposals_tensor, axis=0)
        # Convert to normalized coordinates
        roi_proposals_tensor[:, :, [1, 3]] = roi_proposals_tensor[:, :, [1, 3]] / images.shape[2]
        roi_proposals_tensor[:, :, [2, 4]] = roi_proposals_tensor[:, :, [2, 4]] / images.shape[1]
        assert np.sum(roi_proposals_tensor[:, :, 1:] < 0) == 0 and np.sum(roi_proposals_tensor[:, :, 1:] > 1) == 0
        return images, roi_proposals_tensor

    # def detect_outlier_bbs(self):
    #     training_images = self.dataList[self.trainingImageIndices]
    #     roi_list = []
    #     for img_obj in training_images:
    #         roi_list.append(img_obj.roiMatrix[:, 1:])
    #     roi_list = np.concatenate(roi_list, axis=0)
    #     for coord in [2, 3]:
    #         roi_dim = np.expand_dims(roi_list[:, coord], axis=1)
    #         scaler = StandardScaler()
    #         scaler.fit(roi_dim)
    #         scaled_dim = scaler.transform(roi_dim, copy=True)
    #         print("X")

    def show_image(self, img_obj, scale):
        assert scale in img_obj.imageScales
        img_canvas = np.copy(img_obj.imageScales[scale])
        for roi_row in img_obj.roiMatrix:
            middle_point_x = int(img_canvas.shape[1] * roi_row[1])
            middle_point_y = int(img_canvas.shape[0] * roi_row[2])
            left = int((roi_row[1] - 0.5 * roi_row[3]) * img_canvas.shape[1])
            top = int((roi_row[2] - 0.5 * roi_row[4]) * img_canvas.shape[0])
            right = int((roi_row[1] + 0.5 * roi_row[3]) * img_canvas.shape[1])
            bottom = int((roi_row[2] + 0.5 * roi_row[4]) * img_canvas.shape[0])
            # cv2.circle(img_canvas, (middle_point_x, middle_point_y), radius=3, color=(0, 0, 255))
            cv2.rectangle(img_canvas, (left, top), (right, bottom), color=(0, 0, 255), thickness=1)
            cv2.putText(img_canvas, "{0}".format(int(roi_row[0])),
                        (middle_point_x, middle_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color=(0, 0, 255), thickness=0, lineType=cv2.LINE_AA)
        # cv2.imshow("Image:{0}_Scale:{1}".format(img_obj.imgName, scale), img_canvas)
        # cv2.waitKey()
        cv2.imwrite("Image_{0}.png".format(img_obj.imgName), img_canvas)
        print("X")

    @staticmethod
    def print_img_with_final_rois(img_name, img, roi_matrix, colors):
        img_canvas = np.copy(img)
        for idx, roi_row in enumerate(roi_matrix):
            cv2.rectangle(img_canvas, (roi_row[0], roi_row[1]), (roi_row[2], roi_row[3]), color=colors[idx],
                          thickness=1)
        cv2.imwrite("{0}.png".format(img_name), img_canvas)
