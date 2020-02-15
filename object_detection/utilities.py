import numpy as np


class Utilities:
    def __init__(self):
        pass

    # Vectorized BB vs BB List IoU calculation
    @staticmethod
    def get_iou_with_list(bb, bb_y_matrix):
        assert bb.shape[0] == 4 and bb_y_matrix.shape[1] == 4
        bb_x_matrix = np.repeat(np.expand_dims(bb, axis=0), axis=0, repeats=bb_y_matrix.shape[0])
        intersection_left = np.max(np.stack([bb_x_matrix[:, 0], bb_y_matrix[:, 0]], axis=1), axis=1)
        intersection_top = np.max(np.stack([bb_x_matrix[:, 1], bb_y_matrix[:, 1]], axis=1), axis=1)
        intersection_right = np.min(np.stack([bb_x_matrix[:, 2], bb_y_matrix[:, 2]], axis=1), axis=1)
        intersection_bottom = np.min(np.stack([bb_x_matrix[:, 3], bb_y_matrix[:, 3]], axis=1), axis=1)
        intersection_widths = np.clip(intersection_right - intersection_left, a_min=0, a_max=None)
        intersection_heights = np.clip(intersection_bottom - intersection_top, a_min=0, a_max=None)
        intersection_areas = intersection_widths * intersection_heights
        bb_x_areas = (bb_x_matrix[:, 2] - bb_x_matrix[:, 0]) * (bb_x_matrix[:, 3] - bb_x_matrix[:, 1])
        bb_y_areas = (bb_y_matrix[:, 2] - bb_y_matrix[:, 0]) * (bb_y_matrix[:, 3] - bb_y_matrix[:, 1])
        union_areas = (bb_x_areas + bb_y_areas) - intersection_areas
        iou_vector = intersection_areas * np.reciprocal(union_areas)
        return iou_vector

    # Just simply get iou of two rectangles
    @staticmethod
    def get_iou_of_bbs(bb_x, bb_y):
        bb_x_area = (bb_x[2] - bb_x[0]) * (bb_x[3] - bb_x[1])
        bb_y_area = (bb_y[2] - bb_y[0]) * (bb_y[3] - bb_y[1])
        intersection_left = max(bb_x[0], bb_y[0])
        intersection_right = min(bb_x[2], bb_y[2])
        intersection_top = max(bb_x[1], bb_y[1])
        intersection_bottom = min(bb_x[3], bb_y[3])
        intersection_area = max(intersection_right - intersection_left, 0) \
                            * max(intersection_bottom - intersection_top, 0)
        union_area = bb_x_area + bb_y_area - intersection_area
        iou = intersection_area / union_area
        return iou

    @staticmethod
    def test_vectorized_iou():
        for trial in range(1000):
            print("Trial:{0}".format(trial))
            bb_count = 1000
            bb_left = np.random.uniform(low=0, high=100, size=(bb_count,))
            bb_top = np.random.uniform(low=0, high=100, size=(bb_count,))
            bb_right = np.random.uniform(low=bb_left, high=100, size=(bb_count,))
            bb_bottom = np.random.uniform(low=bb_top, high=100, size=(bb_count,))
            assert np.sum(bb_right >= bb_left) == bb_count and np.sum(bb_bottom >= bb_top) == bb_count
            bb_list = np.stack([bb_left, bb_top, bb_right, bb_bottom], axis=1)
            iou_regular = np.apply_along_axis(func1d=lambda x: Utilities.get_iou_of_bbs(bb_list[0], x), axis=1,
                                              arr=bb_list)
            iou_vectorized = Utilities.get_iou_with_list(bb=bb_list[0], bb_y_matrix=bb_list)
            assert np.allclose(iou_regular, iou_vectorized)
        print("X")


# Utilities.test_vectorized_iou()
