import os
import cv2
import numpy as np
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join

from object_detection.constants import Constants


class DetectionImage(object):
    def __init__(self, img_name, img_arr):
        self.imgName = img_name
        self.imgArr = img_arr
        self.objList = None
        self.imageScales = {}


class ObjectDetectionDataManager:
    def __init__(self, path):
        self.dataPath = path
        self.dataList = None

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
            data_dict[filename].objList = roi_arr
        self.dataList = np.array(list(data_dict.values()))

    def preprocess_data(self):
        for img_obj in self.dataList:
            for img_width in Constants.IMG_WIDTHS:
                new_width = img_width # int(img_obj.imgArr.shape[1] * scale_percent / 100)
                new_height = int((float(img_width) / img_obj.imgArr.shape[1]) * img_obj.imgArr.shape[0])
                img_obj.imageScales[img_width] = cv2.resize(img_obj.imgArr, (new_width, new_height))
        print("X")

    def save_processed_data(self):
        pickle_out_file = open(os.path.join(self.dataPath, "processed_dataset.sav"), "wb")
        pickle.dump(self.dataList, pickle_out_file)
        pickle_out_file.close()

    def load_processed_data(self):
        pickle_in_file = open(os.path.join(self.dataPath, "processed_dataset.sav"), "rb")
        self.dataList = pickle.load(pickle_in_file)
        pickle_in_file.close()

    def show_image(self, img_obj, scale):
        assert scale in img_obj.imageScales
        img_canvas = np.copy(img_obj.imageScales[scale])
        for roi_row in img_obj.objList:
            middle_point_x = int(img_canvas.shape[1] * roi_row[1])
            middle_point_y = int(img_canvas.shape[0] * roi_row[2])
            left = int((roi_row[1] - 0.5 * roi_row[3]) * img_canvas.shape[1])
            top = int((roi_row[2] - 0.5 * roi_row[4]) * img_canvas.shape[0])
            right = int((roi_row[1] + 0.5 * roi_row[3]) * img_canvas.shape[1])
            bottom = int((roi_row[2] + 0.5 * roi_row[4]) * img_canvas.shape[0])
            cv2.circle(img_canvas, (middle_point_x, middle_point_y), radius=3, color=(0, 0, 255))
            cv2.rectangle(img_canvas, (left, top), (right, bottom), color=(0, 0, 255), thickness=1)
        cv2.imshow("Image:{0}_Scale:{1}".format(img_obj.imgName, scale), img_canvas)
        cv2.waitKey()
        print("X")
