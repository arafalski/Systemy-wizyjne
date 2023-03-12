import cv2

img = cv2.imread("qr.jpg")

linear_img = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LINEAR)
nearest_img = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_NEAREST)
area_img = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_AREA)
lanczos_img = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LANCZOS4)

cv2.imshow("original", img)
cv2.imshow("linear", linear_img)
cv2.imshow("nearest", nearest_img)
cv2.imshow("area", area_img)
cv2.imshow("lanczos 4", lanczos_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
