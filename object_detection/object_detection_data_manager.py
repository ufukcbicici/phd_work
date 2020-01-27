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
        for f in img_files:
            filename, file_extension = os.path.splitext(f)
            file_path = os.path.join(self.dataPath, f)
            img_arr = cv2.imread(filename=file_path)
            assert img_arr is not None and len(img_arr.shape) == 3 and img_arr.shape[0] > 0 and img_arr.shape[1] > 0
            img_obj = DetectionImage(img_name=filename, img_arr=img_arr)
            assert filename not in data_dict
            data_dict[filename] = img_obj
        print("X")

