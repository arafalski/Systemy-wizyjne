import cv2

image = cv2.imread("images/forward-1.bmp")

detector = cv2.FastFeatureDetector_create()

keypoints = detector.detect(image)
cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
