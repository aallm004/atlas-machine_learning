#!/usr/bin/env python3
import cv2
import numpy as np


image_damaged = cv2.imread(filename='image_damaged.jpg')

height, width = image_damaged.shape[0], image_damaged.shape[1]

for i in range(height):
    for j in range(width):
        if image_damaged[i, j].sum() > 0:
            image_damaged[i, j] = 0
        else:
            image_damaged[i, j] = [255, 255, 255]

mask = image_damaged
cv2.imwrite('image_damaged.jpg', mask)

cv2.imshow("image mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
