import cv2
import numpy as np

image = cv2.imread("drone_ship.jpg")

edges = cv2.Canny(image, 180, 220)

circles = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=65, minRadius=40, maxRadius=250
)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
