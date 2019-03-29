import os
from enum import Enum
from os import listdir
from os.path import isfile, join
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools


class DatasetTypes(Enum):
    training = 0
    validation = 1
    test = 2


class EggDataset:
    # DATASET_PATH = "//dataset"
    class Classes(Enum):
        background = 0
        egg = 1
        pane = 2

    @staticmethod
    def load_npz(file_name):
        filename = file_name + ".npz"
        try:
            npzfile = np.load(filename)
        except:
            return None
        return npzfile

    @staticmethod
    def save_npz(file_name, arr_dict):
        np.savez(file_name, **arr_dict)

    def __init__(self, window_size, stride, val_ratio=1.0, max_side_length=640):
        self.trainImages = []
        self.trainImageNames = []
        self.validationImages = []
        self.testImages = []
        self.valRatio = val_ratio
        self.trainDataset = None
        self.validationDataset = None
        self.testDataset = None
        self.currentImages = None
        self.currentDataSetType = None
        self.currentIndex = 0
        self.currentDataset = None
        self.isNewEpoch = False
        self.weightDict = {}
        self.windowSize = window_size
        self.stride = stride
        self.currentIndices = []
        self.maxSideLength = max_side_length

    def read_image(self, image_path: str, gray: bool = False) -> np.ndarray:
        if gray:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        longest_side = max(image.shape[0], image.shape[1])
        if longest_side > self.maxSideLength:
            resize_percent = float(self.maxSideLength) / float(longest_side)
            image = cv2.resize(image, None, fx=resize_percent, fy=resize_percent, interpolation=cv2.INTER_AREA)
        return image

    def get_images(self, data_type, image_type):
        path = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), "dataset"), data_type), image_type)
        image_names = {f: os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))}
        images = {train_img_name: self.read_image(image_path=abs_path, gray=image_type == "masks")
                  for train_img_name, abs_path in image_names.items()}
        return images

    def get_cropped_images(self, image, mask):
        # Generate top-left coordinate pairs
        right_boundary = self.windowSize
        bottom_boundary = self.windowSize
        top_coords = []
        left_coords = []
        while True:
            left_coords.append(right_boundary - self.windowSize)
            if right_boundary >= image.shape[1]:
                break
            right_boundary += self.stride
        while True:
            top_coords.append(bottom_boundary - self.windowSize)
            if bottom_boundary >= image.shape[0]:
                break
            bottom_boundary += self.stride
        top_left_coords = list(itertools.product(*[top_coords, left_coords]))
        # Enlarge image with mirroring
        padded_image = np.pad(image, ((0, bottom_boundary - image.shape[0]), (0, right_boundary - image.shape[1]),
                                      (0, 0)), 'symmetric')
        padded_mask = np.pad(mask, ((0, bottom_boundary - mask.shape[0]), (0, right_boundary - image.shape[1]),
                                    (0, 0)), 'symmetric')
        # Crop images
        cropped_imgs = None  # np.zeros(shape=(0, self.windowSize, self.windowSize, 3), dtype=padded_image.dtype)
        cropped_msks = None  # np.zeros(shape=(0, self.windowSize, self.windowSize, 1), dtype=padded_mask.dtype)
        for yx in top_left_coords:
            cropped_img = np.expand_dims(padded_image[yx[0]:yx[0] + self.windowSize, yx[1]:yx[1] + self.windowSize, :],
                                         axis=0)
            cropped_msk = np.expand_dims(padded_mask[yx[0]:yx[0] + self.windowSize, yx[1]:yx[1] + self.windowSize, :],
                                         axis=0)
            cropped_imgs = cropped_img if cropped_imgs is None else np.concatenate((cropped_imgs, cropped_img), axis=0)
            cropped_msks = cropped_msk if cropped_msks is None else np.concatenate((cropped_msks, cropped_msk), axis=0)
        return cropped_imgs, cropped_msks

    def get_cropped_dataset(self, raw_images):
        dataset = None
        for idx, tpl in enumerate(raw_images):
            print(tpl[0].shape)
            cropped_imgs, cropped_msks = self.get_cropped_images(image=tpl[0], mask=tpl[1])
            cropped_data = np.concatenate((cropped_imgs, cropped_msks), axis=3)
            dataset = cropped_data if dataset is None else np.concatenate((dataset, cropped_data), axis=0)
            print(idx)
        return dataset

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
                self.trainImageNames.append(file_name)
        # Test images
        self.validationImages = np.array(self.validationImages)
        self.trainImages = np.array(self.trainImages)
        self.testImages = np.array([(img, None)
                                    for img in self.get_images(data_type="test", image_type="images").values()])
        self.calculate_weighted_map()
        # Build or load the datasets
        # Training
        self.trainDataset = EggDataset.load_npz("cropped_training")
        if self.trainDataset is None:
            self.trainDataset = self.get_cropped_dataset(raw_images=self.trainImages)
            EggDataset.save_npz("cropped_training", {"trainDataset": self.trainDataset})
        else:
            self.trainDataset = self.trainDataset["trainDataset"]
        # Validation
        self.validationDataset = EggDataset.load_npz("cropped_validation")
        if self.validationDataset is None:
            self.validationDataset = self.get_cropped_dataset(raw_images=self.validationImages)
            EggDataset.save_npz("cropped_validation", {"validationDataset": self.validationDataset})
        else:
            self.validationDataset = self.validationDataset["validationDataset"]
        self.set_current_data_set_type(dataset_type=DatasetTypes.training)

    def set_current_data_set_type(self, dataset_type):
        self.currentDataSetType = dataset_type
        if self.currentDataSetType == DatasetTypes.training:
            self.currentImages = self.trainImages
            self.currentDataset = self.trainDataset
        elif self.currentDataSetType == DatasetTypes.test:
            self.currentImages = self.testImages
            self.currentDataset = self.testDataset
        elif self.currentDataSetType == DatasetTypes.validation:
            self.currentImages = self.validationImages
            self.currentDataset = self.validationDataset
        else:
            raise Exception("Unknown dataset type")
        self.reset()

    def calculate_weighted_map(self):
        # pixel_count = 0
        # freq_dict = {EggDataset.Classes.background: 0, EggDataset.Classes.egg: 0, EggDataset.Classes.pane: 0}
        # for idx, tpl in enumerate(self.trainImages):
        #     mask = tpl[1]
        #     # background_count = np.sum(mask == 0)
        #     egg_count = np.sum(mask == 128)
        #     pane_count = np.sum(mask == 255)
        #     background_count = (mask.shape[0] * mask.shape[1]) - (egg_count + pane_count)
        #     freq_dict[EggDataset.Classes.background] += background_count
        #     freq_dict[EggDataset.Classes.egg] += egg_count
        #     freq_dict[EggDataset.Classes.pane] += pane_count
        #     # if (background_count + egg_count + pane_count) != mask.shape[0] * mask.shape[1]:
        #     #     print(self.trainImageNames[idx])
        #     pixel_count += (background_count + egg_count + pane_count)
        # inv_freqs = {k: 1.0 / v for k, v in freq_dict.items()}
        # min_weight = min(inv_freqs.values())
        # self.weightDict = {k: v / min_weight for k, v in inv_freqs.items()}
        self.weightDict = {0: 1.0, 128: 1.0, 255: 1.0}

    def reset(self):
        self.currentIndex = 0
        indices = np.arange(len(self.currentDataset))
        np.random.shuffle(indices)
        # self.currentImages = self.currentImages[indices]
        self.currentDataset = self.currentDataset[indices]
        self.isNewEpoch = False
        print("X")

    def get_next_image(self, make_divisible_to=None):
        image = self.currentImages[self.currentIndex]
        self.currentIndex += 1
        num_of_samples = len(self.currentImages)
        if num_of_samples <= self.currentIndex:
            self.isNewEpoch = True
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        # Pad around the edges of the image
        if make_divisible_to is not None:
            # plt.imshow(image[0])
            # plt.show()
            height_pad = (make_divisible_to - image[0].shape[0] % make_divisible_to) if image[0].shape[0] \
                                                                                        % make_divisible_to != 0 else 0
            width_pad = (make_divisible_to - image[1].shape[1] % make_divisible_to) if image[1].shape[1] \
                                                                                       % make_divisible_to != 0 else 0

            padded_image = np.pad(image[0], ((0, height_pad), (0, width_pad), (0, 0)), 'symmetric')
            padded_mask = np.pad(image[1], ((0, height_pad), (0, width_pad), (0, 0)), 'symmetric')
            assert padded_image.shape[0] == padded_mask.shape[0] and padded_image.shape[1] == padded_mask.shape[1]
            # plt.imshow(padded_image)
            # plt.show()
            final_image = padded_image
            final_mask = padded_mask
        else:
            final_image = image[0]
            final_mask = image[1]
        # Weight array
        weight_img = np.zeros(shape=final_mask.shape, dtype=np.float32)
        weight_img[:] = self.weightDict[0]
        weight_img[final_mask == 128] = self.weightDict[128]
        weight_img[final_mask == 255] = self.weightDict[255]
        # Convert mask array s.t. it contains labels
        temp_mask = np.zeros(shape=final_mask.shape, dtype=final_mask.dtype)
        temp_mask[final_mask == 128] = 1
        temp_mask[final_mask == 255] = 2
        final_mask = temp_mask.astype(np.int32)
        return np.expand_dims(final_image, axis=0), \
               np.expand_dims(final_mask, axis=0), \
               np.expand_dims(weight_img, axis=0)

    def get_next_batch(self, batch_size):
        num_of_samples = self.get_current_sample_count()
        curr_end_index = self.currentIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            curr_end_index = curr_end_index % num_of_samples
            indices_list.extend(self.currentIndices[0:curr_end_index + 1])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        images = self.currentDataset[indices_list][0]
        masks = self.currentDataset[indices_list][1]
        self.currentIndex = self.currentIndex + batch_size
        if num_of_samples <= self.currentIndex:
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        # Weight array
        weight_img = np.zeros(shape=masks.shape, dtype=np.float32)
        weight_img[:] = self.weightDict[0]
        weight_img[masks == 127] = self.weightDict[127]
        weight_img[masks == 255] = self.weightDict[255]
        # Convert mask array s.t. it contains labels
        temp_mask = np.zeros(shape=masks.shape, dtype=masks.dtype)
        temp_mask[masks == 127] = 1
        temp_mask[masks == 255] = 2
        final_mask = temp_mask.astype(np.int32)

    def get_label_count(self):
        return 3

    def get_current_sample_count(self):
        return len(self.currentDataset)
