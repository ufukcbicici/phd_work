import numpy as np


class BBClustering:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self, iou_threshold, max_coverage):
        # Build a total list of all rois
        roi_list = []
        training_objects = self.dataset.dataList[self.dataset.trainingImageIndices]
        for img_obj in training_objects:
            roi_list.append(img_obj.roiMatrix[:, 3:])
        roi_list = np.concatenate(roi_list, axis=0)
        coverage = 0.0
        roi_comparison_list = []
        max_covering_bb_list = []
        for idx, bb in enumerate(roi_list):
            iou_list = np.array(
                sorted([BBClustering.get_iou_of_bbs(bb, roi_list[j]) for j in range(roi_list.shape[0])],
                       reverse=True))
            roi_comparison_list.append(iou_list[iou_list >= iou_threshold])
        while coverage < max_coverage:
            max_covering_bb_idx = np.argmax(np.array([len(iou_list) for iou_list in roi_comparison_list]))
            max_covering_bb_list.append(roi_list[max_covering_bb_idx])
            print("X")

        print("X")

    @staticmethod
    def get_iou_of_bbs(bb_1, bb_2):
        # Put the center of the both rectangles to the origin
        bb_1_coords = [-0.5 * bb_1[0], -0.5 * bb_1[1], 0.5 * bb_1[0], 0.5 * bb_1[1]]
        bb_2_coords = [-0.5 * bb_2[0], -0.5 * bb_2[1], 0.5 * bb_2[0], 0.5 * bb_2[1]]
        intersection_left = max(bb_1_coords[0], bb_2_coords[0])
        intersection_top = max(bb_1_coords[1], bb_2_coords[1])
        intersection_area = (-2.0 * intersection_left) * (-2.0 * intersection_top)
        bb_1_area = (bb_1_coords[2] - bb_1_coords[0]) * (bb_1_coords[3] - bb_1_coords[1])
        bb_2_area = (bb_2_coords[2] - bb_2_coords[0]) * (bb_2_coords[3] - bb_2_coords[1])
        union_area = bb_1_area + bb_2_area - intersection_area
        iou = intersection_area / union_area
        return iou
