import cv2

img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

img_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, 2)
img_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 15, 2)

cv2.imshow('img', img)
cv2.imshow('Adaptive mean', img_mean)
cv2.imshow('Adaptive Gaussian', img_gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()
