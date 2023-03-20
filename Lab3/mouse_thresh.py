import cv2

selected_points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])


image = cv2.imread("parrot.jpg")

g = image[:, :, 1]

cv2.imshow("image", image)
cv2.imshow("g", g)
cv2.setMouseCallback("image", on_mouse)

while True:
    if cv2.waitKey(100) == 27:
        break

    if len(selected_points) == 2:
        (
            _,
            g[
                selected_points[0][1] : selected_points[1][1],
                selected_points[0][0] : selected_points[1][0],
            ],
        ) = cv2.threshold(
            g[
                selected_points[0][1] : selected_points[1][1],
                selected_points[0][0] : selected_points[1][0],
            ],
            200,
            255,
            cv2.THRESH_BINARY,
        )

        cv2.imshow("image", image)
        cv2.imshow("g", g)

cv2.destroyAllWindows()
