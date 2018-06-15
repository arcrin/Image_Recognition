import cv2
import numpy as np


img = cv2.imread('cpu_01.jpg')
rows, cols = img.shape[:2]

kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
kernel_3x3 = np.ones((3, 3), np.float32) / 9.0 # Divide by 9 to normalize the kernel
kernel_5x5 = np.ones((5, 5), np.float32) / 25.0 # Divide by 25 to normalize the kernel

cv2.imshow("Original", img)

# value -1 is to maintain source image depth
output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow("Identity filter", output)

output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow("3x3 filter", output)

output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow("5x5 filter", output)

output = cv2.blur(img, (3, 3))
cv2.imshow("OpenCV 3x3 blur", output)

cv2.waitKey()