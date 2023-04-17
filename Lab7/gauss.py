import cv2
import numpy as np


def get_frame(cap):
    ret, frame = cap.read()
    if ret == False:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(0)

current_window = "current image"
mask = "mask"

cv2.namedWindow(current_window)
cv2.namedWindow(mask)

cv2.createTrackbar("threshold", mask, 70, 255, lambda x: None)

back_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    frame = get_frame(cap)
    frame_mask = back_sub.apply(frame)

    frame_mask = cv2.morphologyEx(
        frame_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    threshold = cv2.getTrackbarPos("threshold", mask)
    _, frame_mask = cv2.threshold(frame_mask, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow(current_window, frame)
    cv2.imshow(mask, frame_mask)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
