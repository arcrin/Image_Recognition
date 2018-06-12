import cv2
import numpy as np


input_img = cv2.imread("cpu_01.jpg")
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray_img, None)

gray_img = cv2.drawKeypoints(gray_img, keypoints,
                              None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.namedWindow('SIFT features', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SIFT features', 600, 600)
cv2.imshow('SIFT features', gray_img)
cv2.waitKey()