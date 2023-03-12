import cv2
import numpy as np


def empty_callback(value):
    pass


img1 = cv2.imread('red.png')
img2 = cv2.imread('logo.png')
current_img = np.zeros(img1.shape, dtype=np.uint8)
cv2.namedWindow("image")

cv2.createTrackbar("alpha", "image", 0, 100, empty_callback)

dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

while True:
    cv2.imshow("image", current_img)

    key_code = cv2.waitKey(10)
    if key_code == ord("q"):
        break

    alpha = cv2.getTrackbarPos("alpha", "image") / 100
    current_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

cv2.destroyAllWindows()
