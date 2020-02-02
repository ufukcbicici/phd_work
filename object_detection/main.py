import os
import numpy as np

from object_detection.object_detection_data_manager import ObjectDetectionDataManager


def main():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager(path=path)
    # dataset.read_data()
    # dataset.preprocess_data()
    # dataset.save_processed_data()
    dataset.load_processed_data()
    dataset.show_image(img_obj=dataset.dataList[10], scale=320)


if __name__ == "__main__":
    main()
