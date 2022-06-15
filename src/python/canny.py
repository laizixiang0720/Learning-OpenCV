import cv2
import numpy as np

img = cv2.imread("../resources/images/statue_small.jpg")

cv2.imshow("canny", cv2.Canny(img, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()
