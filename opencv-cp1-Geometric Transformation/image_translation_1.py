import cv2
import numpy as np


img = cv2.imread('empire.jpg')
num_rows, num_columns = img.shape[:2]

translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
img_translation = cv2.warpAffine(img, translation_matrix, (num_columns, num_rows))
cv2.imshow('Original', img)
cv2.imshow('Translation', img_translation)
cv2.waitKey()