import os
import numpy as np

from object_detection.object_detection_data_manager import ObjectDetectionDataManager


def main():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager(path=path)
    dataset.read_data()


if __name__ == "__main__":
    main()