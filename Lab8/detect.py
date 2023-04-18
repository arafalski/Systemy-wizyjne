import cv2
import numpy as np
from time import time

image = cv2.imread("images/forward-1.bmp")

fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()
sift = cv2.SIFT_create()

tick = time()
keypoints_fast = fast.detect(image)
print(f"FAST num of keypoints: {len(keypoints_fast)}")
print(f"FAST time: {time() - tick:.4f} s")
print()

image_fast = np.zeros_like(image)
cv2.drawKeypoints(
    image, keypoints_fast, image_fast, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

tick = time()
keypoints_orb = orb.detect(image)
print(f"ORB num of keypoints: {len(keypoints_orb)}")
print(f"OBR time: {time() - tick:.4f} s")
print()

image_orb = np.zeros_like(image)
cv2.drawKeypoints(
    image, keypoints_orb, image_orb, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

tick = time()
keypoints_sift = orb.detect(image)
print(f"SIFT num of keypoints: {len(keypoints_sift)}")
print(f"SIFT time: {time() - tick:.4f} s")
print()

image_sift = np.zeros_like(image)
cv2.drawKeypoints(
    image, keypoints_sift, image_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("fast", image_fast)
cv2.imshow("orb", image_orb)
cv2.imshow("sift", image_sift)

cv2.waitKey(0)
cv2.destroyAllWindows()
