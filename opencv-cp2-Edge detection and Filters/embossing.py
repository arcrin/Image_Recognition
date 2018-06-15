import cv2
import numpy as np


input_img = cv2.imread('defect.jpg')

img_emboss_input = cv2.resize(input_img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
# generating the kernels
emboss_kernel_1 = np.array([[0, -1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])
emboss_kernel_2 = np.array([[-1, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 1]])
emboss_kernel_3 = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])

emboss_kernel_4 = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])


emboss_kernel_5 = np.array([[1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

# converting the image to grayscale
gray_img = cv2.cvtColor(img_emboss_input, cv2.COLOR_BGR2GRAY)

# applying the kernels to the gray scale image and adding the offset to produce the shadow
output_1 = cv2.filter2D(gray_img, -1, emboss_kernel_1) + 128
output_2 = cv2.filter2D(gray_img, -1, emboss_kernel_2) + 128
output_3 = cv2.filter2D(gray_img, -1, emboss_kernel_3) + 128
output_4 = cv2.filter2D(gray_img, -1, emboss_kernel_4) + 128
output_5 = cv2.filter2D(gray_img, -1, emboss_kernel_5) + 128

cv2.imwrite('emboss_cpu.jpg', output_5)

# cv2.imshow('Input', img_emboss_input)
# cv2.imshow('Embossing - South West', output_1)
# cv2.imshow('Embossing - South East', output_2)
# cv2.imshow('Embossing - North West', output_3)
# cv2.imshow('Embossing - Test', output_4)
cv2.imshow('Embossing - 5x5', output_5)


cv2.waitKey(0)