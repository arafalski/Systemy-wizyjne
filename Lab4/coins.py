import cv2
import numpy as np

image = cv2.imread("coins.jpg")

blur = cv2.GaussianBlur(image, (17, 17), 0)
edges = cv2.Canny(blur, 100, 180)
# cv2.imshow('blur', cv2.resize(blur, dsize=None, fx=0.8, fy=0.8))
# cv2.imshow('edges', cv2.resize(edges, dsize=None, fx=0.8, fy=0.8))

circles = cv2.HoughCircles(
    edges, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=20, minRadius=50, maxRadius=100
)
circles = np.uint16(np.around(circles))

coins_sum = 0.0
for i in circles[0, :]:
    if i[2] > 70:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 4)
        coins_sum += 1.0
    else:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 4)
        coins_sum += 0.1

print(f"Sum: {coins_sum:.2f} PLN")
cv2.imshow("image", cv2.resize(image, dsize=None, fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.destroyAllWindows()
