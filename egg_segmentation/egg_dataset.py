import os
from enum import Enum
from os import listdir
from os.path import isfile, join
from typing import Optional, Tuple

import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class DatasetTypes(Enum):
    training = 0
    validation = 1
    test = 2


class EggDataset:
    # DATASET_PATH = "//dataset"

    def __init__(self, val_ratio=0.8):
        self.trainImages = []
        self.validationImages = []
        self.testImages = []
        self.valRatio = val_ratio
        self.trainDataset = None
        self.validationDataset = None
        self.testDataset = None
        self.currentImages = None
        self.currentDataSetType = None
        self.currentIndex = 0
        self.isNewEpoch = False

    def read_image(self, image_path: str, gray: bool = False) -> np.ndarray:
        if gray:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_images(self, data_type, image_type):
        path = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), "dataset"), data_type), image_type)
        image_names = {f: os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))}
        images = {train_img_name: self.read_image(image_path=abs_path, gray=image_type == "masks")
                  for train_img_name, abs_path in image_names.items()}
        return images

    def load_dataset(self):
        # Get files for training
        all_imgs = self.get_images(data_type="train", image_type="images")
        all_masks = self.get_images(data_type="train", image_type="masks")
        # Create training and validation datasets
        all_file_names = np.array([f_name for f_name in all_imgs.keys()])
        val_set_size = int(all_file_names.shape[0] * (1.0 - self.valRatio))
        val_indices = np.random.choice(all_file_names.shape[0], val_set_size, False)
        val_file_names = set(all_file_names[val_indices].tolist())
        assert len(val_file_names) == val_set_size
        for file_name in all_file_names:
            all_masks[file_name] = np.reshape(a=all_masks[file_name],
                                              newshape=(all_masks[file_name].shape[0],
                                                        all_masks[file_name].shape[1], 1))
            if file_name in val_file_names:
                self.validationImages.append((all_imgs[file_name], all_masks[file_name]))
            else:
                self.trainImages.append((all_imgs[file_name], all_masks[file_name]))
        # Test images
        self.validationImages = np.array(self.validationImages)
        self.trainImages = np.array(self.trainImages)
        self.testImages = np.array([(img, None)
                                    for img in self.get_images(data_type="test", image_type="images").values()])
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)

    def set_current_data_set_type(self, dataset_type):
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == DatasetTypes.training:
            self.currentImages = self.trainImages
        elif self.currentDataSetType == DatasetTypes.test:
            self.currentImages = self.testImages
        elif self.currentDataSetType == DatasetTypes.validation:
            self.currentImages = self.validationImages
        else:
            raise Exception("Unknown dataset type")
        self.reset()

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(len(self.currentImages))
        np.random.shuffle(indices)
        self.currentImages = self.currentImages[indices]
        self.isNewEpoch = False
        print("X")

    def get_next_image(self):
        image = self.currentImages[self.currentIndex]
        self.currentIndex += 1
        num_of_samples = len(self.currentImages)
        if num_of_samples <= self.currentIndex:
            self.isNewEpoch = True
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        return image
