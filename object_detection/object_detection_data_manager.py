import os
import cv2
import numpy as np
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from sklearn.preprocessing import StandardScaler


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

    def process_data(self, iou_threshold, max_coverage, test_ratio=0.15):
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
                                           max_coverage=max_coverage)

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
            #

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

            #  Get medoid distribution
            found_positive_rois = 0

            # while found_positive_rois < positive_roi_count:
            #     x_coords = np.random.uniform(low=0, high=selected_scale - selected_roi[0], size=(roi_sample_count, ))
            #     y_coords = np.random.uniform(low=0, high=img_height - selected_roi[1], size=(roi_sample_count,))
            #     roi_list = np.stack([x_coords, y_coords], axis=1)

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
    def print_img_with_final_rois(img_name, img, roi_matrix):
        img_canvas = np.copy(img)
        for roi_row in roi_matrix:
            cv2.rectangle(img_canvas, (roi_row[1], roi_row[2]), (roi_row[3], roi_row[4]), color=(0, 0, 255),
                          thickness=1)
        cv2.imwrite("{0}.png".format(img_name), img_canvas)
