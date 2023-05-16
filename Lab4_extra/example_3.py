import cv2

image_1 = cv2.imread("zad_stop.jpg")
image_2 = cv2.imread("zad_STOP_2.jpg")

image_1 = cv2.resize(image_1, (int(500 / image_1.shape[0] * image_1.shape[1]), 500))
image_2 = cv2.resize(image_2, (int(500 / image_2.shape[0] * image_2.shape[1]), 500))

blur_1 = cv2.GaussianBlur(image_1, (5, 5), 0)
blur_2 = cv2.GaussianBlur(image_2, (5, 5), 0)

edges_1 = cv2.Canny(blur_1, 50, 200)
edges_2 = cv2.Canny(blur_2, 100, 200)

contours_1, hierarchy_1 = cv2.findContours(
    edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(image_1, contours_1, -1, (0,255,0), 1)

cv2.imshow("image_1", image_1)
cv2.imshow("image_2", image_2)
cv2.imshow("edges_1", edges_1)
cv2.imshow("edges_2", edges_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
