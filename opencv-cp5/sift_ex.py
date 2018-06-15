import cv2
import numpy as np
import matplotlib.pyplot as plt



input_img = cv2.imread("cpu_erode.jpg")
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray_img, None)

# size_list = sorted([point.size for point in keypoints])
#
# fig, ax = plt.subplots()
# ax.plot(size_list)
# plt.show()

new_points = [point for point in keypoints if point.size > 11.0]

gray_img = cv2.drawKeypoints(gray_img, new_points,
                              None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.namedWindow('SIFT features', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('SIFT features', 600, 600)
cv2.imshow('SIFT features', gray_img)
cv2.waitKey()