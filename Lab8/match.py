import cv2

image1 = cv2.imread("images/rotate-1.bmp")
image2 = cv2.imread("images/rotate-6.bmp")

orb = cv2.ORB_create()

kp1, desc1 = orb.detectAndCompute(image1, None)
kp2, desc2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher()
matches = bf.match(desc1, desc2)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:20]

image_matches = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imshow("matches", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
