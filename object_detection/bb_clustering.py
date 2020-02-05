import numpy as np

from auxillary.general_utility_funcs import UtilityFuncs


class BBClustering:

    @staticmethod
    def run(training_objects, iou_threshold, max_coverage, trial_count=10):
        # Build a total list of all rois
        global medoids, cost
        roi_list = []
        for img_obj in training_objects:
            roi_list.append(img_obj.roiMatrix[:, 3:])
        roi_list = np.concatenate(roi_list, axis=0)
        # Build the distance matrix
        iou_similarity_matrix = np.zeros(shape=(roi_list.shape[0], roi_list.shape[0]))
        for idx, bb in enumerate(roi_list):
            iou_vec = BBClustering.get_iou_of_bbs_vec(bb_x=bb, bb_y_list=roi_list)
            iou_similarity_matrix[idx, :] = iou_vec
        iou_distance_matrix = 1.0 - iou_similarity_matrix
        # Apply clustering until enough coverage is reached.
        curr_coverage = 0.0
        curr_medoid_count = 1
        best_cost = 1e10
        best_medoids = None
        for trial_id in range(trial_count):
            while curr_coverage < max_coverage:
                medoids, cost = BBClustering.k_medoids(medoid_count=curr_medoid_count,
                                                       iou_distance_matrix=iou_distance_matrix)
                curr_coverage = BBClustering.get_coverage(medoids=medoids, iou_distance_matrix=iou_distance_matrix,
                                                          iou_threshold=iou_threshold)
                print("Current Medoid Count={0} Current Coverage={1}".format(curr_medoid_count, curr_coverage))
                curr_medoid_count += 1
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids
        medoid_rois = roi_list[best_medoids]
        return medoid_rois

        # The following is for testing
        # medoid_rois = roi_list[medoids]
        # medoid_similarity_matrix = np.zeros(shape=(roi_list.shape[0], medoid_rois.shape[0]))
        # for idx, bb in enumerate(roi_list):
        #     iou_vec = BBClustering.get_iou_of_bbs_vec(bb_x=bb, bb_y_list=medoid_rois)
        #     medoid_similarity_matrix[idx, :] = iou_vec
        # medoid_distance_matrix = 1.0 - medoid_similarity_matrix
        # cost = BBClustering.get_configuration_cost(distances_to_medoids=medoid_distance_matrix)
        # assert np.allclose(cost, best_cost)
        # print("X")

    @staticmethod
    def get_configuration_cost(distances_to_medoids):
        medoid_assignments = np.argmin(distances_to_medoids, axis=1)
        total_cost = 0.0
        for medoid_id in range(distances_to_medoids.shape[1]):
            distances = distances_to_medoids[medoid_assignments == medoid_id, medoid_id]
            total_cost += np.sum(distances)
        return total_cost

    @staticmethod
    def k_medoids(medoid_count, iou_distance_matrix, max_num_of_iterations=100):
        curr_medoids = np.random.choice(iou_distance_matrix.shape[0], medoid_count, replace=False)
        best_cost = BBClustering.get_configuration_cost(iou_distance_matrix[:, curr_medoids])
        for iteration_id in range(max_num_of_iterations):
            medoids_changed = False
            medoid_member_pairs = UtilityFuncs.get_cartesian_product(
                [np.arange(medoid_count), np.arange(iou_distance_matrix.shape[0])])
            print("*******Iteration:{0} Medoids:{1} Cost:{2}*******".format(iteration_id, curr_medoids, best_cost))
            for mo_pair in medoid_member_pairs:
                medoid_idx = mo_pair[0]
                object_idx = mo_pair[1]
                if np.any(curr_medoids == object_idx):
                    continue
                new_medoids = np.copy(curr_medoids)
                new_medoids[medoid_idx] = object_idx
                cost = BBClustering.get_configuration_cost(iou_distance_matrix[:, new_medoids])
                if cost < best_cost:
                    best_cost = cost
                    curr_medoids = new_medoids
                    print("*******Iteration:{0} Medoids:{1} Cost:{2}*******".format(iteration_id, curr_medoids,
                                                                                    best_cost))
                    medoids_changed = True
            if not medoids_changed:
                break
        return curr_medoids, best_cost

    @staticmethod
    def get_coverage(medoids, iou_distance_matrix, iou_threshold):
        distances_to_medoids = iou_distance_matrix[:, medoids]
        min_distances = np.min(distances_to_medoids, axis=1)
        similarities = 1.0 - min_distances
        coverage = np.sum(similarities >= iou_threshold) / float(iou_distance_matrix.shape[0])
        return coverage

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
