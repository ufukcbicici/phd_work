import os
import numpy as np

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from object_detection.fast_rcnn import FastRcnn
from object_detection.fast_rcnn_with_bb_regression import FastRcnnWithBBRegression
from object_detection.object_detection_data_manager import ObjectDetectionDataManager

# Global Detector Object
global_detector = None


# Fast R-CNN Module Entry Points
def create_dataset():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager(path=path)
    dataset.read_data()
    dataset.process_data(iou_threshold=Constants.MAX_IOU_DISTANCE,
                         max_coverage=Constants.MAX_INCLUSIVENESS_BB,
                         test_ratio=Constants.TEST_RATIO,
                         max_num_of_medoids=Constants.MAX_MEDOID_COUNT)
    dataset.save_processed_data()


def load_dataset():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.join(os.path.join(curr_path, ".."), "data"), "tuborgData")
    dataset = ObjectDetectionDataManager.load_processed_data(data_path=path)
    dataset.calculate_label_distribution()
    dataset.filter_uncovered_images()
    return dataset


def load_fast_rcnn_detector():
    dataset = load_dataset()
    global_detector = FastRcnn(roi_list=dataset.medoidRois,
                               background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    global_detector.build_network()
    global_detector.load_model(iteration=5000)
    # global_detector.calculate_accuracy_on_image(img=dataset.dataList[0].imgArr,
    #                                             ground_truth_list=dataset.dataList[0].roiMatrix)
    predictions = global_detector.detect_single_image(original_img=dataset.dataList[0].imgArr)
    dataset.print_img_with_final_rois(img_name="{0}_{1}".format(dataset.dataList[0].imgName, 5000),
                                      img=dataset.dataList[0].imgArr,
                                      roi_matrix=predictions[:, 2:].astype(np.int32),
                                      colors=[(0, 255, 0)] * predictions.shape[0])


def train_fast_rcnn_detector():
    dataset = load_dataset()
    detector = FastRcnn(roi_list=dataset.medoidRois,
                        background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    detector.build_network()
    detector.train(dataset=dataset)


def train_fast_rcnn_detector_with_bb_regression():
    dataset = load_dataset()
    detector = FastRcnnWithBBRegression(roi_list=dataset.medoidRois,
                                        background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    detector.build_network()
    detector.train(dataset=dataset)


def image_detection_test():
    dataset = load_dataset()
    detector = FastRcnn(roi_list=dataset.medoidRois,
                        background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    detector.build_network()
    detect_image(detector, dataset.dataList[0].imgArr)


def detect_image(detector, img):
    detector.detect_single_image(img=img)


def main():
    # create_dataset()
    # image_detection_test()
    # dataset = load_dataset()
    # dataset.create_image_batch(
    #     batch_size=Constants.IMAGE_COUNT_PER_BATCH,
    #     roi_sample_count=Constants.ROI_SAMPLE_COUNT_PER_IMAGE,
    #     positive_sample_ratio=Constants.POSITIVE_SAMPLE_RATIO_PER_IMAGE)

    train_fast_rcnn_detector_with_bb_regression()
    # load_fast_rcnn_detector()

    # train_fast_rcnn_detector_with_bb_regression()
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
