import cv2
import json
import numpy as np

import matplotlib.pyplot as plt

with open('planogram.json') as f:
	planogram_dict = json.load(f)



scene_width = planogram_dict["DOLAP"]["W"]
scene_height = planogram_dict["DOLAP"]["H"]
scene_x = planogram_dict["DOLAP"]["X"]
scene_y = planogram_dict["DOLAP"]["Y"]

image = np.zeros((scene_height, scene_width, 3))

image = cv2.rectangle(image, (scene_x, scene_y), (scene_x+scene_width, scene_y+scene_height), (1,1,1), thickness=-1)

for shelf in planogram_dict["DOLAP"]["RAFLAR"]:
	shelf_width = shelf["W"]
	shelf_height = shelf["H"]
	shelf_x = shelf["X"]
	shelf_y = shelf["Y"]
	image = cv2.rectangle(image, (shelf_x, shelf_y), (shelf_x+shelf_width, shelf_y+shelf_height), (0.5,0.5,0.5), thickness=-1)

	for product in shelf["ÜRÜNLER"]:
		product_width = product["W"]
		product_height = product["H"]
		product_x = product["X"]
		product_y = product["Y"]
		product_type = product["SINIF TİPİ"]
		image = cv2.rectangle(image, (product_x, product_y), (product_x+product_width, product_y+product_height), (0.5,0,0), thickness=-1)
		image = cv2.putText(image, product_type, (product_x+10, product_y+product_height//2), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False) 


plt.imshow(image)
plt.show() 
