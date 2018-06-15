import cv2


img = cv2.imread("defect.jpg")
img_scaled = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
cv2.imshow("Scaling - Linear Interpolation", img_scaled)
img_scaled = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Scaling - Cubic Interpolation", img_scaled)
img_scaled = cv2.resize(img, (450, 400), interpolation=cv2.INTER_AREA)
cv2.imshow("Scaling - Skewed Size", img_scaled)
cv2.waitKey()