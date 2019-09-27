import os
import numpy as np
import pandas as pd


class BoundingBox:
    def __init__(self, top, left, bottom, right, class_id, class_name, confidence, image_name):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.classId = class_id
        self.className = class_name
        self.confidence = confidence
        self.imageName = image_name


class MapCalculator:
    def __init__(self, ground_truth_csv, predictions_csv):
        self.classFrequencies = {}
        self.groundTruthCsv = pd.read_csv(ground_truth_csv)
        self.predictionsCsv = pd.read_csv(predictions_csv)
        self.groundTruthBbDict = self.read_bounding_box_csv(csv_file=self.groundTruthCsv)
        self.predictionsBbDict = self.read_bounding_box_csv(csv_file=self.predictionsCsv)

    def read_bounding_box_csv(self, csv_file):
        bb_dict = {}
        for index, row in csv_file.iterrows():
            # filename	x_min	y_min	x_max	y_max	class
            bb = BoundingBox(left=row["x_min"], right=row["x_max"],
                             top=row["y_min"], bottom=row["y_max"],
                             image_name=row["filename"], class_name=row["class"],
                             confidence=1.0, class_id=None)
            if bb.imageName not in bb_dict:
                bb_dict[bb.imageName] = []
            bb_dict[bb.imageName].append(bb)
        return bb_dict

    def analyze_class_distributions(self):
        for img_name, bb_list in self.groundTruthBbDict.items():
            for bb in bb_list:
                if bb.className not in self.classFrequencies:
                    self.classFrequencies[bb.className] = 0
                self.classFrequencies[bb.className] += 1

    def calculate(self, threshold_iou):
        self.analyze_class_distributions()
        print("X")
        # For each class
        for class_name in self.classFrequencies.keys():
            # Get all predictions on all images
            class_predictions_dict = {}
            for img_name, predicted_bb_list in self.predictionsBbDict.items():
                # Get ground truth bounding boxes, for the current class, on the current image
                img_ground_truth_bb_list = [gt_bb for gt_bb in self.groundTruthBbDict[img_name]
                                            if gt_bb.className == class_name]
                # Get all predictions on this image for the current class
                img_predictions_bb_list = [prediction_bb for prediction_bb in self.predictionsBbDict[img_name]
                                           if prediction_bb.className == class_name]
                print("X")


# ground_truth_csv_path = os.path.join("C:", "Users", "ufuk.bicici", "Documents", "preds", "ground_truth.csv")
# predictions_csv_path = os.path.join("C:", "Users", "ufuk.bicici", "Documents", "preds", "predictions.csv")

ground_truth_csv_path = "C://Users//ufuk.bicici//Documents//preds//ground_truth.csv"
predictions_csv_path = "C://Users//ufuk.bicici//Documents//preds//predictions.csv"

map_calculator = MapCalculator(ground_truth_csv=ground_truth_csv_path, predictions_csv=predictions_csv_path)
map_calculator.calculate(threshold_iou=0.5)
