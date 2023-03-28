import cv2

image = cv2.imread("not_bad.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image_thresh = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)
image_thresh = cv2.morphologyEx(
    image_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
)
image_thresh = cv2.morphologyEx(
    image_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
)

contours, hierarchy = cv2.findContours(
    image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
cv2.drawContours(image, contours, -1, (0, 255, 0), cv2.FILLED)

cv2.imshow("image", cv2.resize(image, None, fx=0.3, fy=0.3))
cv2.imshow("thresh", cv2.resize(image_thresh, None, fx=0.3, fy=0.3))
cv2.waitKey(0)
cv2.destroyAllWindows()
