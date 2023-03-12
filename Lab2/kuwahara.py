import cv2
import numpy as np


def create_img_with_padding(img: np.ndarray, border_size: int):
    img_border_shape = (img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size)

    img_border = np.zeros(img_border_shape, dtype=np.uint8)

    img_border[border_size:-border_size, border_size:-border_size] = img

    # Left
    img_border[border_size:-border_size, 0:border_size] = img_border[
        border_size:-border_size, border_size
    ].reshape((len(img), 1))
    # Right
    img_border[
        border_size:-border_size, -border_size : img_border.shape[1]
    ] = img_border[border_size:-border_size, -border_size - 1].reshape((len(img), 1))
    # Top
    img_border[0:border_size] = img_border[border_size]
    # Bottom
    img_border[-border_size : len(img_border)] = img_border[-border_size - 1]

    return img_border


def apply_kuwahara(image: np.ndarray, window_size: int):
    border_size = window_size // 2
    image_border = create_img_with_padding(image, border_size)
    image_new = np.zeros_like(image)

    for y in range(image_new.shape[0]):
        for x in range(image_new.shape[1]):
            window = image_border[y : y + window_size, x : x + window_size]
            regions = [
                window[0 : border_size + 1, 0 : border_size + 1],
                window[border_size:window_size, 0 : border_size + 1],
                window[0 : border_size + 1, border_size:window_size],
                window[border_size:window_size, border_size:window_size],
            ]

            mean_std = [cv2.meanStdDev(r) for r in regions]
            image_new[y, x] = min(mean_std, key=lambda x: x[1])[0]

    return image_new


img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

img2 = img.copy()
for y in range(0, img2.shape[0], 3):
    for x in range(0, img2.shape[1], 3):
        img2[y, x] = 255

img_kuwahara = apply_kuwahara(img2, 5)

cv2.imshow("original", img2)
cv2.imshow("kuwahara", img_kuwahara)
cv2.waitKey(0)
cv2.destroyAllWindows()
