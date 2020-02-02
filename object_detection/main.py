import os
import numpy as np

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from object_detection.object_detection_data_manager import ObjectDetectionDataManager


def main():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    # dataset = ObjectDetectionDataManager(path=path)
    # dataset.read_data()
    # dataset.preprocess_data()
    # dataset.save_processed_data()
    dataset = ObjectDetectionDataManager.load_processed_data(data_path=path)
    # dataset.show_image(img_obj=dataset.dataList[10], scale=640)
    bb_clustering_algorithm = BBClustering(dataset=dataset)
    bb_clustering_algorithm.run(iou_threshold=Constants.POSITIVE_IOU_THRESHOLD,
                                max_coverage=Constants.MAX_INCLUSIVENESS_BB)


if __name__ == "__main__":
    main()
