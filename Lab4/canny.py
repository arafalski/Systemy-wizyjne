import cv2


def empty_callback(value):
    pass


image = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("canny")
cv2.createTrackbar("lower", "canny", 0, 255, empty_callback)
cv2.createTrackbar("upper", "canny", 0, 255, empty_callback)

while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
        break

    lower = cv2.getTrackbarPos("lower", "canny")
    upper = cv2.getTrackbarPos("upper", "canny")

    canny = cv2.Canny(image, lower, upper)
    cv2.imshow("canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
