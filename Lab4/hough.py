import cv2
import numpy as np

image = cv2.imread("shapes.jpg")
image = cv2.resize(image, dsize=None, fx=0.7, fy=0.7)

edges = cv2.Canny(image, 160, 200)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


circles = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40, minRadius=20, maxRadius=250
)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
