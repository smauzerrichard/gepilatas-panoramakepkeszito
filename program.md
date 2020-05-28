import cv2

import numpy as np

img = cv2.imread('kep1.jpg', 0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img1 = cv2.imread('kep2.jpg', cv2.IMREAD_GRAYSCALE)

result = cv2.matchTemplate(gray_img, img1, cv2.TM_CCOEFF_NORMED)

loc = np.where(result >= 0.8)

print(loc)

cv2.imshow('image', img)

cv2.imshow('result', result)

cv2.waitKey(0)

cv2.destroyAllWindows()
