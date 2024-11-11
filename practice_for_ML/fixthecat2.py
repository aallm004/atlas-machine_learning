#!/usr/bin/env python3
import cv2
import numpy as np

img = cv2.imread('image_damaged.jpg')

mask = cv2.imread('cat_mask.png', 0)

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite('cat_inpainted.png', dst)

