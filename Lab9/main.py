import cv2
import numpy as np

selected_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global selected_points
        selected_points.append((x, y))


image_to_find = cv2.imread("objects_to_find.jpg")
image_to_find = cv2.resize(image_to_find, None, fx=0.25, fy=0.25)

cv2.namedWindow("image ori")
cv2.setMouseCallback("image ori", mouse_callback)

cv2.namedWindow("image cut")

while True:
    cv2.imshow("image ori", image_to_find)

    if len(selected_points) == 4:
        x, y, w, h = cv2.boundingRect(np.array(selected_points))
        cv2.imshow("image cut", image_to_find[y:y + h, x:x + w])
        selected_points.clear()

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
