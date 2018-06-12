import cv2
gray_img = cv2.imread("empire.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale", gray_img)
cv2.imwrite('output.jpg', gray_img)
cv2.waitKey()