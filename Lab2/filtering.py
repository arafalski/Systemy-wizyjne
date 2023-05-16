import cv2


def empty_callback(value):
    pass


img = cv2.imread("lenna_salt_and_pepper.bmp", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("image")
cv2.namedWindow("blur")
cv2.namedWindow("median")
cv2.namedWindow("gauss")

cv2.createTrackbar("kernel size", "image", 0, 50, empty_callback)

while True:
    key_code = cv2.waitKey(10)
    if key_code == ord("q"):
        break

    kernel_size = cv2.getTrackbarPos("kernel size", "image")
    img_blur = cv2.blur(img, (2 * kernel_size + 1, 2 * kernel_size + 1))
    img_median = cv2.medianBlur(img, 2 * kernel_size + 1)
    img_gauss = cv2.GaussianBlur(img, (2 * kernel_size + 1, 2 * kernel_size + 1), 0)

    cv2.imshow("image", img)
    cv2.imshow("blur", img_blur)
    cv2.imshow("median", img_median)
    cv2.imshow("gauss", img_gauss)

cv2.destroyAllWindows()
