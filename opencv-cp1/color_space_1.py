import cv2


img = cv2.imread('empire.jpg')
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Grayscale imgae", gray_img)
cv2.imshow("HSV imgae", hsv_img)
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey()