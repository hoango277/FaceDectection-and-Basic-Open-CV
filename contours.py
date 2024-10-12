import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')
cv.imshow('cat', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

canny = cv.Canny(gray, 100, 200)
cv.imshow('canny', canny)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('blank', blank)
contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cv.drawContours(blank,contours, -1, (0,0,255),1)
cv.imshow('contours', blank)


cv.waitKey(0)