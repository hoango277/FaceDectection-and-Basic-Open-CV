import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')

blank = np.zeros(img.shape[:2], dtype='uint8')

cv.rectangle(blank,(0,0), [img.shape[1]//2, img.shape[0]//2], [255,255,255], thickness=-1)
cv.imshow('img',blank )

masked = cv.bitwise_and(img, img, mask=blank)
cv.imshow('masked',masked)

cv.waitKey(0)