from email.mime import image

import cv2 as cv
import numpy as np
img = cv.imread("Photos/cat.jpg")
cv.imshow("Cat", img)
# translation
def translate(img, x, y):
    transMat =np.float32([[1,0,x],[0,1,y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)


translated = translate(img,100,100)
cv.imshow("Translated", translated)
# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (int(width//2), int(height//2))

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)
    return cv.warpAffine(img, rotMat, dimension)

rotated = rotate(img, 45)
cv.imshow("Rotated", rotated)

# Resizing
resized = cv.resize(img, (500, 500))
cv.imshow("Resized", resized)
# Flipping
flip = cv.flip(img, 1)
cv.imshow("Flip", flip)
# Cropping
cropped = img[100:200, 100:200]
cv.imshow("Cropped", cropped)




cv.waitKey(0)

