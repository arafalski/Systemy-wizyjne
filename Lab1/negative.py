import cv2

img = cv2.imread("lenna.png")

negative = 255 - img

cv2.imshow('img', img)
cv2.imshow('negative', negative)
cv2.waitKey(0)
cv2.destroyAllWindows()
