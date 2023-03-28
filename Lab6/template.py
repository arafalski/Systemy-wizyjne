import cv2

image = cv2.imread("lenna.png")
template = cv2.imread("lenna_template.png")
w, h = template.shape[1], template.shape[0]

res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)

cv2.imshow("image", image)
cv2.imshow("template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
