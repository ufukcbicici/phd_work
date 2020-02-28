import cv2
import json
import numpy as np

import matplotlib.pyplot as plt


class PlanogramVisualizer:

    def __init__(self, json_path):
        with open(json_path) as f:
            self.planogram_dict = json.load(f)

        self.scene_width = self.planogram_dict["DOLAP"]["W"]
        self.scene_height = self.planogram_dict["DOLAP"]["H"]
        self.scene_x = self.planogram_dict["DOLAP"]["X"]
        self.scene_y = self.planogram_dict["DOLAP"]["Y"]
        self.productColorMapping = {}
        self.colorSet = set()
        self.planogramImage = self.get_planogram_image()

    def pick_color_for_product(self):
        while True:
            random_color = tuple(np.random.uniform(low=0, high=1, size=(3,)).tolist())
            if random_color not in self.colorSet:
                self.colorSet.add(random_color)
                return random_color

    def get_planogram_image(self):
        image = np.zeros((self.scene_height, self.scene_width, 3))
        # Get product color mappings
        for shelf in self.planogram_dict["DOLAP"]["RAFLAR"]:
            for product in shelf["URUNLER"]:
                product_type = product["SINIF"]
                if product_type not in self.productColorMapping:
                    product_color = self.pick_color_for_product()
                    self.productColorMapping[product_type] = product_color

        for shelf in self.planogram_dict["DOLAP"]["RAFLAR"]:
            shelf_width = shelf["W"]
            shelf_height = shelf["H"]
            shelf_x = shelf["X"]
            shelf_y = shelf["Y"]
            image = cv2.rectangle(image, (shelf_x, shelf_y), (shelf_x + shelf_width, shelf_y + shelf_height),
                                  (0.5, 0.5, 0.5), thickness=-1)

            for product in shelf["URUNLER"]:
                product_width = product["W"]
                product_height = product["H"]
                product_x = product["X"]
                product_y = product["Y"]
                product_type = product["SINIF"]
                image = cv2.rectangle(image, (product_x, product_y),
                                      (product_x + product_width, product_y + product_height),
                                      self.productColorMapping[product_type],
                                      thickness=-1)
                # image = cv2.putText(image, product_type, (product_x + 10, product_y + product_height // 2),
                #                     cv2.FONT_HERSHEY_SIMPLEX,
                #                     0.5, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
        return image

    def homographic_transform(self, image, source_points, target_points):
        h, _ = cv2.findHomography(source_points, target_points)
        image = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
        return image

    # def get_planogram_compliance(self, detection_json):
    #     img_data = json.loads(x)


if __name__ == "__main__":
    planogram = PlanogramVisualizer("planogram.json")


    # image = planogram.get_planogram_image()
    # plt.imshow(image)
    # plt.show()
    #
    # source_points = np.array([[planogram.scene_x, planogram.scene_y],
    #                           [planogram.scene_x + planogram.scene_width, planogram.scene_y],
    #                           [planogram.scene_x + planogram.scene_width, planogram.scene_y + planogram.scene_height],
    #                           [planogram.scene_x, planogram.scene_y + planogram.scene_height]])
    #
    # target_points = np.array([[planogram.scene_x, planogram.scene_y],
    #                           [planogram.scene_x + planogram.scene_width, planogram.scene_y],
    #                           [planogram.scene_x + 10 + planogram.scene_width,
    #                            planogram.scene_y + planogram.scene_height - 10],
    #                           [planogram.scene_x + 10, planogram.scene_y + planogram.scene_height - 10]])
    # image = planogram.homographic_transform(image, source_points, target_points)
    #
    # plt.imshow(image)
    # plt.show()
