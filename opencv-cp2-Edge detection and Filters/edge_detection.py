import cv2
import numpy as np


img = cv2.imread('cpu_01.jpg', cv2.IMREAD_GRAYSCALE)
edge_sharpening_kernel = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0

edge_sharpen_image = cv2.filter2D(img, -1, edge_sharpening_kernel)

rows, cols = edge_sharpen_image.shape



# It is used depth of cv2.CV_64F
sobel_horizontal = cv2.Sobel(edge_sharpen_image, cv2.CV_64F, 1, 0, ksize=5)

# Kernel size can be: 1, 3, 5, or 7
sobel_vertial = cv2.Sobel(edge_sharpen_image, cv2.CV_64F, 0, 1, ksize=5)

laplacian_img = cv2.Laplacian(edge_sharpen_image, cv2.CV_64F)
canny_img = cv2.Canny(edge_sharpen_image, 50, 249)
# cv2.imwrite('cpu_canny.jpg', canny_img)

# cv2.imwrite('cpu_edge_detect.jpg', sobel_vertial)

cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertial)
cv2.imshow('Laplacian', laplacian_img)
cv2.imshow('Canny', canny_img)


cv2.waitKey(0)

