import cv2
import numpy as np
from time import perf_counter

img = np.zeros((10000, 10000), dtype=np.uint8)

tick = perf_counter()
cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LINEAR)
t_linear = perf_counter() - tick

tick = perf_counter()
cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_NEAREST)
t_nearest = perf_counter() - tick

tick = perf_counter()
cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_AREA)
t_area = perf_counter() - tick

tick = perf_counter()
cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LANCZOS4)
t_lanczos = perf_counter() - tick

print(f"Linear: {t_linear}")
print(f"Neares: {t_nearest}")
print(f"Area: {t_area}")
print(f"Lanczos4: {t_lanczos}")
