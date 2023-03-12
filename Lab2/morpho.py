import cv2
import numpy as np


def empty_callback(value):
    pass


img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("threshold")
cv2.namedWindow("erosion")
cv2.namedWindow("dilation")
cv2.namedWindow("opening")
cv2.namedWindow("closing")

_, img_thresh = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

cv2.createTrackbar("kernel size", "threshold", 0, 50, empty_callback)

while True:
    key_code = cv2.waitKey(10)
    if key_code == ord("q"):
        break

    kernel_size = cv2.getTrackbarPos("kernel size", "threshold")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    img_erosion = cv2.erode(img_thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(img_thresh, kernel, iterations=1)
    img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("threshold", img_thresh)
    cv2.imshow("erosion", img_erosion)
    cv2.imshow("dilation", img_dilation)
    cv2.imshow("opening", img_opening)
    cv2.imshow("closing", img_closing)

cv2.destroyAllWindows()
