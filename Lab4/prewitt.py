import cv2
import numpy as np

image = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

prewitt_x = cv2.filter2D(image, cv2.CV_32F, kernel_prewitt_x)
prewitt_y = cv2.filter2D(image, cv2.CV_32F, kernel_prewitt_y)
sobel_x = cv2.filter2D(image, cv2.CV_32F, kernel_sobel_x)
sobel_y = cv2.filter2D(image, cv2.CV_32F, kernel_sobel_y)

cv2.imshow("image", image)
cv2.imshow("Prewitt x", prewitt_x)
cv2.imshow("Prewitt y", prewitt_y)
cv2.imshow("Sobel x", sobel_x)
cv2.imshow("Sobel y", sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
