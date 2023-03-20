import cv2
import numpy as np

selected_points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])


gallery = cv2.imread("gallery.png")
pug = cv2.imread("pug.png")

cv2.imshow("gallery", gallery)
cv2.setMouseCallback("gallery", on_mouse)

while True:
    cv2.imshow("gallery", gallery)

    if cv2.waitKey(100) == 27:
        break

    if len(selected_points) == 4:
        source_points = np.float32(
            [[0, 0], [pug.shape[1], 0], [pug.shape[1], pug.shape[0]], [0, pug.shape[0]]]
        )

        matrix = cv2.getPerspectiveTransform(source_points, np.float32(selected_points))
        pug_result = cv2.warpPerspective(
            pug, matrix, (gallery.shape[1], gallery.shape[0])
        )

        _, pug_th = cv2.threshold(pug_result, 0, 255, cv2.THRESH_BINARY)
        pug_closed = cv2.morphologyEx(
            pug_th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        gallery[pug_closed != 0] = 0
        gallery += pug_result

        cv2.imshow("gallery", gallery)

cv2.waitKey()
