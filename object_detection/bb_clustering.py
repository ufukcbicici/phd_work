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
            # iou_list = np.array(
            #     sorted([BBClustering.get_iou_of_bbs(bb, roi_list[j]) for j in range(roi_list.shape[0])],
            #            reverse=True))
            iou_list = np.array(sorted(BBClustering.get_iou_of_bbs_vec(bb_x=bb, bb_y_list=roi_list), reverse=True))
            # assert np.allclose(iou_list, iou_list_vec)
            roi_comparison_list.append(iou_list[iou_list >= iou_threshold])
        while coverage < max_coverage:
            max_covering_bb_idx = int(np.argmax(np.array([len(iou_list) for iou_list in roi_comparison_list])))
            max_covering_bb_list.append(roi_list[max_covering_bb_idx])
            roi_comparison_list.pop(max_covering_bb_idx)
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

    @staticmethod
    def get_iou_of_bbs_vec(bb_x, bb_y_list):
        bb_x_coords = np.expand_dims(np.array([-0.5 * bb_x[0], -0.5 * bb_x[1], 0.5 * bb_x[0], 0.5 * bb_x[1]]), axis=0)
        bb_x_coords = np.repeat(bb_x_coords, axis=0, repeats=bb_y_list.shape[0])
        bb_y_coord_matrix = np.stack(
            [-0.5 * bb_y_list[:, 0], -0.5 * bb_y_list[:, 1], 0.5 * bb_y_list[:, 0], 0.5 * bb_y_list[:, 1]], axis=1)
        intersection_left = np.max(np.stack([bb_x_coords[:, 0], bb_y_coord_matrix[:, 0]], axis=1), axis=1)
        intersection_top = np.max(np.stack([bb_x_coords[:, 1], bb_y_coord_matrix[:, 1]], axis=1), axis=1)
        intersection_areas = (-2.0 * intersection_left) * (-2.0 * intersection_top)
        bb_x_area = np.array((bb_x_coords[0, 2] - bb_x_coords[0, 0]) * (bb_x_coords[0, 3] - bb_x_coords[0, 1]))
        bb_y_areas = (bb_y_coord_matrix[:, 2] - bb_y_coord_matrix[:, 0]) * \
                     (bb_y_coord_matrix[:, 3] - bb_y_coord_matrix[:, 1])
        union_areas = np.expand_dims(bb_x_area, axis=0) + bb_y_areas - intersection_areas
        iou_list = intersection_areas * np.reciprocal(union_areas)
        return iou_list
