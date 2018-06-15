import cv2
import numpy as np


img = cv2.imread('cpu_01.jpg')
cv2.imshow("Original", img)


# generate kernel
sharpening_kernel_1 = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
sharpening_kernel_2 = np.array([[1, 1, 1],
                                [1, -7, 1]])
sharpening_kernel_3 = np.array([[-1, -1, -1, -1, -1],
                                [-1, 2, 2, 2, -1],
                                [-1, 2, 8, 2, -1],
                                [-1, 2, 2, 2, -1],
                                [-1, -1, -1, -1, -1]]) / 8.0

output_1 = cv2.filter2D(img, -1, sharpening_kernel_1)
output_2 = cv2.filter2D(img, -1, sharpening_kernel_2)
output_3 = cv2.filter2D(img, -1, sharpening_kernel_3)


cv2.imshow("Sharpening", output_1)
cv2.imshow("Excessive Sharpening", output_2)
cv2.imshow("Edge Sharpening", output_3)

cv2.waitKey(0)

