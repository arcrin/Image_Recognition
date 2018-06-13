import cv2
import numpy as np


img = cv2.imread('cpu_01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


surf = cv2.xfeatures2d.SURF_create(hessianThreshold=15000)

kp, des = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(img, kp, None, (0, 255), 4)


# cv2.namedWindow('SURF features', 0)
# cv2.resizeWindow('SURF features', 500, 500)
cv2.imshow('SURF features', img)
cv2.waitKey()