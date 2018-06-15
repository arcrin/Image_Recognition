import cv2
gray_image = cv2.imread("cpu_01.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale", gray_image)
cv2.imwrite("images/output.jpg", gray_image)
cv2.waitKey()