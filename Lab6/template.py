import cv2
import numpy as np

image = cv2.imread("lenna.png")
template = cv2.imread("lenna_template.png")
w, h = template.shape[1], template.shape[0]

res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.imshow("template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
