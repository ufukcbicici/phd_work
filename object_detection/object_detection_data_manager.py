import os
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


class DetectionImage:
    def __init__(self, img_name, img_arr):
        self.imgName = img_name
        self.imgArr = img_arr
        self.objList = None


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
        print("X")

            # img_arr = cv2.imread(filename=file_path)
            # assert img_arr is not None and len(img_arr.shape) == 3 and img_arr.shape[0] > 0 and img_arr.shape[1] > 0
            # img_obj = DetectionImage(img_name=filename, img_arr=img_arr)
            # assert filename not in data_dict
            # data_dict[filename] = img_obj

        print("X")

