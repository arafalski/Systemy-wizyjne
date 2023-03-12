import cv2


def empty_callback(value):
    pass


img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
current_img = img.copy()
cv2.namedWindow("image")

cv2.createTrackbar("threshold", "image", 0, 255, empty_callback)
cv2.createTrackbar("thresh type", "image", 0, 4, empty_callback)

while True:
    cv2.imshow("image", current_img)

    key_code = cv2.waitKey(10)
    if key_code == ord("q"):
        break

    threshold = cv2.getTrackbarPos("threshold", "image")
    thresh_type_num = cv2.getTrackbarPos("thresh type", "image")
    thresh_type = cv2.THRESH_BINARY

    if thresh_type_num == 1:
        thresh_type = cv2.THRESH_BINARY_INV
    elif thresh_type_num == 2:
        thresh_type = cv2.THRESH_TRUNC
    elif thresh_type_num == 3:
        thresh_type = cv2.THRESH_TOZERO
    elif thresh_type_num == 4:
        thresh_type = cv2.THRESH_TOZERO_INV

    _, current_img = cv2.threshold(img, threshold, 255, thresh_type)

cv2.destroyAllWindows()
