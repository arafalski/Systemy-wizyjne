import cv2
import numpy as np

image = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

kernel_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

sobel_x = cv2.filter2D(image, cv2.CV_32F, kernel_sobel_x)
sobel_y = cv2.filter2D(image, cv2.CV_32F, kernel_sobel_y)

mod = np.sqrt(sobel_x**2 + sobel_y**2)
mod = (mod * 255 / np.amax(mod)).astype(np.uint8)

cv2.imshow("image", image)
cv2.imshow("mod", mod)

cv2.waitKey(0)
cv2.destroyAllWindows()
