import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")

image = cv2.imread("parrot.jpg", cv2.IMREAD_GRAYSCALE)
image_eq = cv2.equalizeHist(image)

plt.hist(image_eq.flatten(), bins=256, range=(0, 256))
plt.show()

cv2.imshow("ori", image)
cv2.imshow("flatten", image_eq)
cv2.waitKey()
cv2.destroyAllWindows()
