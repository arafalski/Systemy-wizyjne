import cv2
import numpy as np


def get_frame(cap):
    ret, frame = cap.read()
    if ret == False:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(0)

background_window = "background image"
current_window = "current image"
foreground_window = "foreground image"

cv2.namedWindow(background_window)
cv2.namedWindow(current_window)
cv2.namedWindow(foreground_window)

cv2.createTrackbar("threshold", foreground_window, 70, 255, lambda x: None)

background_image = get_frame(cap)

while True:
    current_image = get_frame(cap)
    cv2.imshow(current_window, current_image)
    cv2.imshow(background_window, background_image)

    foreground_image = cv2.absdiff(background_image, current_image)

    foreground_image = cv2.morphologyEx(
        foreground_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    threshold = cv2.getTrackbarPos("threshold", foreground_window)
    _, foreground_image = cv2.threshold(
        foreground_image, threshold, 255, cv2.THRESH_BINARY
    )

    cv2.imshow(foreground_window, foreground_image)

    background_image[background_image < current_image] += 1
    background_image[background_image > current_image] -= 1

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord("q"):
        break
    elif key_pressed == ord("a"):
        background_image = get_frame(cap)
        is_background_captured = background_image is not None

cap.release()
cv2.destroyAllWindows()
