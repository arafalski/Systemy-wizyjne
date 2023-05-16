import cv2
import numpy as np
from time import perf_counter

img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

img2 = img.copy()
for y in range(0, img2.shape[0], 3):
    for x in range(0, img2.shape[1], 3):
        img2[y, x] = 255

img_blur1 = img2.copy()
tick = perf_counter()
for y in range(1, img_blur1.shape[0]):
    for x in range(1, img_blur1.shape[1]):
        img_blur1[y, x] = np.mean(img2[y - 1 : y + 2, x - 1 : x + 2])
t_for = perf_counter() - tick

tick = perf_counter()
img_blur2 = cv2.blur(img2, (3, 3))
t_opencv = perf_counter() - tick

tick = perf_counter()
kernel = np.ones((3, 3), dtype=np.uint8) / 9
cv2.filter2D(img2, -1, kernel)
t_filter2D = perf_counter() - tick

print(f"For: {t_for} s")
print(f"Opencv: {t_opencv} s")
print(f"Filter 2D: {t_filter2D} s")

cv2.imshow("image", img2)
cv2.imshow("blur1", img_blur1)
cv2.waitKey(0)
cv2.destroyAllWindows()
