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
cv2.drawKeypoints(image, keypoints_fast, image_fast)

tick = time()
keypoints_orb = orb.detect(image)
print(f"ORB num of keypoints: {len(keypoints_orb)}")
print(f"OBR time: {time() - tick:.4f} s")
print()

image_orb = np.zeros_like(image)
cv2.drawKeypoints(image, keypoints_orb, image_orb)

tick = time()
keypoints_sift = orb.detect(image)
print(f"SIFT num of keypoints: {len(keypoints_sift)}")
print(f"SIFT time: {time() - tick:.4f} s")
print()

image_sift = np.zeros_like(image)
cv2.drawKeypoints(image, keypoints_sift, image_sift)

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
tick = time()
keypoints_fast, desc_brief = brief.compute(image, keypoints_fast)
print(f"BRIEF time: {time() - tick:.4f} s")
print(desc_brief.shape)
print()

tick = time()
keypoints_orb, desc_orb = orb.compute(image, keypoints_orb)
print(f"ORB time: {time() - tick:.4f} s")
print(desc_orb.shape)
print()

tick = time()
keypoints_sift, desc_sift = sift.compute(image, keypoints_sift)
print(f"SIFT time: {time() - tick:.4f} s")
print(desc_sift.shape)
print()

cv2.imshow("fast", image_fast)
cv2.imshow("orb", image_orb)
cv2.imshow("sift", image_sift)

cv2.waitKey(0)
cv2.destroyAllWindows()
