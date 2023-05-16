import cv2
import numpy as np

selected_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global selected_points
        selected_points.append((x, y))


image_to_find = cv2.imread("objects_to_find.jpg")
image_to_find = cv2.resize(image_to_find, None, fx=0.25, fy=0.25)

image_to_match = cv2.imread("objects_template.jpg")
image_to_match = cv2.resize(image_to_match, None, fx=0.25, fy=0.25)

detector = cv2.SIFT_create()

cv2.namedWindow("image ori")
cv2.setMouseCallback("image ori", mouse_callback)

while True:
    cv2.imshow("image ori", image_to_find)

    if len(selected_points) == 4:
        x, y, w, h = cv2.boundingRect(np.array(selected_points))
        image_cut = image_to_find[y:y + h, x:x + w]

        kp_cut, desc_cut = detector.detectAndCompute(image_cut, None)
        kp_image, desc_image = detector.detectAndCompute(image_to_match, None)

        bf = cv2.BFMatcher()
        matches = bf.match(desc_cut, desc_image)

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:20]

        image_matches = cv2.drawMatches(
            image_cut,
            kp_cut,
            image_to_match,
            kp_image,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("matches", image_matches)
        selected_points.clear()

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
