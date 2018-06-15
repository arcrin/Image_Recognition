import cv2
import numpy as np

img = cv2.imread('cpu_01.jpg', 0) # reads image with gray scale

# equalize the histogram of the input image
histeq = cv2.equalizeHist(img)

cv2.imshow('Input', img)
cv2.imshow('Histogram equalize', histeq)
cv2.waitKey()
