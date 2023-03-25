import cv2

selected_points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])


image = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("image", image)
cv2.setMouseCallback("image", on_mouse)


while True:
    if cv2.waitKey(100) == ord("q"):
        break

    if len(selected_points) == 2:
        first_x, first_y = selected_points[0]
        second_x, second_y = selected_points[1]

        image[first_y:second_y, first_x:second_x] = cv2.Canny(
            image[first_y:second_y, first_x:second_x], 100, 200
        )

        selected_points.clear()
        cv2.imshow("image", image)

cv2.destroyAllWindows()
