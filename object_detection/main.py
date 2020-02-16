import os
import numpy as np

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from object_detection.object_detection_data_manager import ObjectDetectionDataManager


# Fast R-CNN Module Entry Points
def create_dataset(iou_threshold, max_coverage, test_ratio):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager(path=path)
    dataset.read_data()
    dataset.process_data(iou_threshold=iou_threshold, max_coverage=max_coverage, test_ratio=test_ratio)
    dataset.save_processed_data()


def load_dataset():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager.load_processed_data(data_path=path)
    return dataset


def main():
    dataset = load_dataset()
    dataset.create_image_batch(batch_size=3, positive_iou_threshold=Constants.POSITIVE_IOU_THRESHOLD,
                               roi_sample_count=100, positive_sample_ratio=0.25)

    # create_dataset(iou_threshold=Constants.POSITIVE_IOU_THRESHOLD, max_coverage=Constants.MAX_INCLUSIVENESS_BB,
    #                test_ratio=0.15)

    # curr_path = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")

    # dataset = ObjectDetectionDataManager(path=path)
    # dataset.read_data()
    # dataset.process_data()
    # dataset.save_processed_data()

    # dataset = ObjectDetectionDataManager.load_processed_data(data_path=path)
    # # dataset.show_image(img_obj=dataset.dataList[10], scale=640)
    # bb_clustering_algorithm = BBClustering(dataset=dataset)
    # bb_clustering_algorithm.run(iou_threshold=Constants.POSITIVE_IOU_THRESHOLD,
    #                             max_coverage=Constants.MAX_INCLUSIVENESS_BB)


if __name__ == "__main__":
    main()
