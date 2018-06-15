import cv2
import numpy as np


img = cv2.imread("cpu_01.jpg")
num_rows, num_cols = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
translation_matrix = np.float32([[1, 0, int(0.5 * num_cols)], [0, 1, int(0.5 * num_rows)]])


img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_cols, 2 * num_rows))
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey()