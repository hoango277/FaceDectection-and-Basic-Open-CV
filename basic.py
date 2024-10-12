import cv2 as cv

img = cv.imread("Photos/cat.jpg")
# covert image to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('test', gray)

# blur
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)
# canny edges
canny = cv.Canny(img, 125, 175)
cv.imshow('canny', canny)
# dilating the image

dilated = cv.dilate(canny, (7, 7), iterations=1)
cv.imshow('dilated', dilated)

# eroding
erode = cv.erode(dilated, (5, 5), iterations=1)
cv.imshow('erode', erode)



cv.waitKey(0)