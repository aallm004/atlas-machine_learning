import cv2
import numpy as np
 
# reading the damaged imagege img_daman = cv2.imread(filename=r"image_damaged.png")
 
# get the shape of the image
height, width = damaged_img.shape[0], damaged_img.shape[1]
 
# Converting all pixels greater than zero to black while black becomes white
for i in range(height):
    for j in range(width):
        if damaged_img[i, j].sum() > 0:
            damaged_img[i, j] = 0
        else:
            damaged_img[i, j] = [255, 255, 255]
 
# saving the mask 
mask = damaged_img
cv2.imwrite('mask.jpg', mask)
 
# displaying mask
cv2.imshow("damaged image mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


