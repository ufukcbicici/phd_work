import os
import numpy as np

from object_detection.bb_clustering import BBClustering
from object_detection.constants import Constants
from object_detection.fast_rcnn import FastRcnn
from object_detection.fast_rcnn_with_bb_regression import FastRcnnWithBBRegression
from object_detection.object_detection_data_manager import ObjectDetectionDataManager

# Global Detector Object
from object_detection.planogram_measurement import PlanogramCompliance

global_detector = None
planogram_compliance = None


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
    global global_detector
    global planogram_compliance
    planogram_compliance = PlanogramCompliance("planogram.json")
    dataset = load_dataset()
    global_detector = FastRcnn(roi_list=dataset.medoidRois,
                               background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    global_detector.build_network()
    global_detector.load_model(iteration=Constants.MODEL_ID)


def load_fast_rcnn_detector_with_bb_regression():
    global global_detector
    global planogram_compliance
    planogram_compliance = PlanogramCompliance("planogram.json")
    dataset = load_dataset()
    global_detector = FastRcnnWithBBRegression(roi_list=dataset.medoidRois,
                                               background_label=dataset.backgroundLabel, class_count=dataset.classCount)
    global_detector.build_network()
    global_detector.load_model(iteration=Constants.MODEL_ID)


def test_on_all_images(type="train"):
    dataset = load_dataset()
    load_fast_rcnn_detector_with_bb_regression()
    os.mkdir("groundtruths")
    os.mkdir("detections")
    if type == "test":
        data_list = dataset.dataList[dataset.testImageIndices]
    elif type == "train":
        data_list = dataset.dataList[dataset.trainingImageIndices]
    else:
        data_list = dataset.dataList
    for img_obj in data_list:
        print("New Image")
        global_detector.calculate_accuracy_on_image(img_name=img_obj.imgName, img=img_obj.imgArr,
                                                    roi_matrix=img_obj.roiMatrix, type=type)


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


# def image_detection_test():
#     dataset = load_dataset()
#     detector = FastRcnn(roi_list=dataset.medoidRois,
#                         background_label=dataset.backgroundLabel, class_count=dataset.classCount)
#     detector.build_network()
#     detect_image(detector, dataset.dataList[0].imgArr)


def detect_image(img, upper_left, upper_right, bottom_left, bottom_right):
    global global_detector
    json_file = global_detector.detect_single_image_json(original_img=img)
    planogram_compliance.get_planogram_compliance(detection_json=json_file,
                                                  upper_left=upper_left, upper_right=upper_right,
                                                  bottom_left=bottom_left, bottom_right=bottom_right)
    return json_file


def sample_call():
    dataset = load_dataset()
    load_fast_rcnn_detector_with_bb_regression()
    upper_left = np.array([0, 0])
    upper_right = np.array([0, dataset.dataList[0].imgArr.shape[1]])
    bottom_left = np.array([dataset.dataList[0].imgArr.shape[0], 0])
    bottom_right = np.array([dataset.dataList[0].imgArr.shape[0], dataset.dataList[0].imgArr.shape[1]])
    json_file = detect_image(img=dataset.dataList[0].imgArr, upper_left=upper_left,
                             upper_right=upper_right, bottom_left=bottom_left, bottom_right=bottom_right)


def main():
    # sample_call()
    test_on_all_images(type="train")
    # print("X")

    # create_dataset()
    # image_detection_test()
    # dataset = load_dataset()
    # dataset.create_image_batch(
    #     batch_size=Constants.IMAGE_COUNT_PER_BATCH,
    #     roi_sample_count=Constants.ROI_SAMPLE_COUNT_PER_IMAGE,
    #     positive_sample_ratio=Constants.POSITIVE_SAMPLE_RATIO_PER_IMAGE)

    # train_fast_rcnn_detector_with_bb_regression()
    # load_fast_rcnn_detector()
    # test_on_all_images()
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
