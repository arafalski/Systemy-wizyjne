import cv2
import numpy as np

image = cv2.imread("fruit.jpg")

edges = cv2.Canny(image, 180, 220)

circles = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=100, maxRadius=170
)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    width = round(i[2] / np.sqrt(2))

    hsv_area = cv2.cvtColor(
        image[i[1] - width : i[1] + width, i[0] - width : i[0] + width],
        cv2.COLOR_BGR2HSV,
    )
    mean_h = np.mean(hsv_area[:, :, 0])

    if mean_h < 20:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 4)
    elif mean_h > 25:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 4)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
