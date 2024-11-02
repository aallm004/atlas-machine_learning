#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('image_damaged.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#use biharmonic equation to repair image
mask = create_mask_from_black(gray, threshold=10)
A = get_neighborhood_matrix(mask)
result = biharmonic_equation(A, mask, gray)
result = np.clip(result, 0, 255).astype(np.uint8)
show_results(image, result)
cv2.imwrite('image_repaired.jpg', result)

