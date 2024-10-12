import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')
# Averaging
average = cv.blur(img, (5,5))
cv.imshow('average', average)
# Median Blur
median = cv.medianBlur(img,3)
cv.imshow('median', median)
# Bilateral
bilateral = cv.bilateralFilter(img,9,75,75)
cv.imshow('bilateral', bilateral)

cv.waitKey(0)