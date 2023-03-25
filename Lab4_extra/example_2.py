import cv2
import numpy as np

image_src = cv2.imread("zad_kółka_1.png")
image = image_src.copy()

background_mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
image[background_mask == 255] = 0

blue_mask = cv2.inRange(image, (200, 100, 0), (255, 200, 0))
red_mask = cv2.inRange(image, (0, 0, 200), (0, 0, 255))

blue_mask = cv2.morphologyEx(
    blue_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
)
red_mask = cv2.morphologyEx(
    red_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
)

blue_edges = cv2.Canny(blue_mask, 100, 200)
red_edges = cv2.Canny(red_mask, 100, 200)

blue_circles = cv2.HoughCircles(
    blue_edges,
    cv2.HOUGH_GRADIENT,
    1,
    50,
    param1=100,
    param2=20,
    minRadius=50,
    maxRadius=100,
)
blue_circles = np.uint16(np.around(blue_circles))

for i in blue_circles[0, :]:
    cv2.circle(image_src, (i[0], i[1]), i[2], (0, 0, 255), 4)

red_circles = cv2.HoughCircles(
    red_edges,
    cv2.HOUGH_GRADIENT,
    1,
    50,
    param1=100,
    param2=20,
    minRadius=50,
    maxRadius=100,
)
red_circles = np.uint16(np.around(red_circles))

for i in red_circles[0, :]:
    cv2.circle(image_src, (i[0], i[1]), i[2], (255, 0, 0), 4)

cv2.imshow("image", image_src)
cv2.waitKey(0)
cv2.destroyAllWindows()
